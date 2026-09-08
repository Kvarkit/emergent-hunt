"""Frozen greedy-policy EH-INT-r0.1 export. No optimization or RNG sampling."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch
from .train import SlotSender, SlotReceiver, BagReceiver, mlp, features
from .intervention import build_rows, control_summary

# architecture tag -> (sender constructor, receiver constructor(head)).
# This is a REGISTRY, not a fallback chain (nadir-codex #25734, gate 1): every
# tag train.run() can write under --architecture must have an explicit entry
# here, and an unrecognized or missing tag is a hard error, not a silent
# generic-mlp guess. Before this was a registry, bag_receiver had no entry
# and fell through to the generic-mlp branch by default: load_state_dict
# raised on mismatched keys (net.0.weight vs 0.weight -- melioralab-agent
# #25638), but a DIFFERENT unlucky shape collision could have loaded wrong
# weights silently instead of raising. A closed registry can't do that: any
# tag not listed below is refused before torch ever sees the state_dict.
_NETWORK_BY_ARCHITECTURE = {
    'mlp': (lambda: mlp(6, 16), lambda head: mlp(19, 6 if head == 'factorized' else 9)),
    'slots': (lambda: SlotSender(), lambda head: SlotReceiver()),
    'receiver_slots': (lambda: mlp(6, 16), lambda head: SlotReceiver()),
    'bag_receiver': (lambda: mlp(6, 16), lambda head: BagReceiver()),
}


def build_networks(architecture, head):
    """Registry lookup (see _NETWORK_BY_ARCHITECTURE). Raises ValueError on
    any tag not explicitly listed -- never guesses a generic shape."""
    if architecture not in _NETWORK_BY_ARCHITECTURE:
        raise ValueError(
            f'unknown architecture tag {architecture!r}; diagnose_checkpoint '
            f'has no registered reconstructor for it (known: '
            f'{sorted(_NETWORK_BY_ARCHITECTURE)}). Add one instead of '
            f'guessing a shape -- a wrong guess that happens to load without '
            f'raising is worse than one that raises (#25734).')
    build_sender, build_receiver = _NETWORK_BY_ARCHITECTURE[architecture]
    return build_sender(), build_receiver(head)


@torch.no_grad()
def fixed_probe_signature(sender, receiver, head, n=3):
    """Identity check (nadir-codex #25734, gate 2): a hash of this network's
    greedy outputs over the full corpus for fixed inputs, independent of and
    stronger than 'load_state_dict did not raise'. Two receivers can be
    shape-compatible (same parameter names and tensor shapes) while wiring
    the same weights to different inputs -- load_state_dict succeeds on
    both, but their outputs on the same probe differ. That is exactly what
    this signature is for: it is deterministic (eval mode, argmax, no
    sampling) and independent of the intervention corpus construction, so it
    is not circular with what diagnose() itself reports.
    """
    sender.eval()
    receiver.eval()
    outputs = []
    for p in range(n):
        for d in range(n):
            sent = tuple(sender(features(torch.tensor([[p, d]]), 3)).view(2, 8).argmax(-1).tolist())
            wire = features(torch.tensor([sent]), 8)
            for t in range(n):
                logits = receiver(torch.cat((features(torch.tensor([[t]]), n), wire), -1))
                action = (logits.view(2, 3).argmax(-1).tolist() if head == 'factorized'
                          else [logits.argmax(-1).item()])
                outputs.append((p, d, t, sent, tuple(action)))
    return hashlib.sha256(repr(outputs).encode()).hexdigest()


def diagnose(path):
    payload = Path(path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    saved = torch.load(path, map_location='cpu', weights_only=True)
    head = saved['head']
    architecture = saved['architecture']
    sender, receiver = build_networks(architecture, head)
    sender.load_state_dict(saved['sender'])
    receiver.load_state_dict(saved['receiver'])
    sender.eval()
    receiver.eval()
    network_signature = fixed_probe_signature(sender, receiver, head)

    @torch.no_grad()
    def send(p, d):
        if saved['mode'] == 'no_message':
            return ()
        return tuple(sender(features(torch.tensor([[p, d]]), 3)).view(2, 8).argmax(-1).tolist())

    @torch.no_grad()
    def receive(t, message):
        wire = features(torch.tensor([message]), 8) if message else torch.zeros(1, 16)
        logits = receiver(torch.cat((features(torch.tensor([[t]]), 3), wire), -1))
        if head == 'factorized':
            return (*logits.view(2, 3).argmax(-1).tolist(), t)
        action = logits.argmax(-1).item()
        return (action//3, action%3, t)

    rows = build_rows(seed=saved['seed'], policies=[('learned_greedy', send, receive)], checkpoint=digest)
    for row in rows:
        row['training_split'] = saved['by']
        row['architecture'] = saved['architecture']
        row['training_reward'] = saved['reward_kind']
    assert len(rows) == 162
    assert all(r['sent_before'] == r['sent_after'] for r in rows if r['factor'] == 't')
    return {'checkpoint_sha256': digest, 'network_signature': network_signature,
            'summary': control_summary(rows), 'rows': rows}


class IdentityMismatch(Exception):
    """A checkpoint's actual sha256 or network_signature doesn't match the
    value it was expected to match. Raised, not just reported -- nadir-codex
    #25752/#25800: a forged-decoder checkpoint producing a DIFFERENT
    network_signature only proves the two functions differ. It does not by
    itself make a QA pipeline fail closed unless something actually compares
    against an expected value and refuses to proceed on mismatch. This
    exception, and verify()'s nonzero CLI exit, are that refusal."""


def verify(path, expect_checkpoint_sha256=None, expect_network_signature=None):
    """diagnose(path), then fail closed (raise IdentityMismatch) if either
    expected value is given and doesn't match. A caller who wants "this
    checkpoint must still be the specific artifact/decoder we pinned, not
    merely A/some valid checkpoint" calls this instead of diagnose()."""
    report = diagnose(path)
    if expect_checkpoint_sha256 is not None and report['checkpoint_sha256'] != expect_checkpoint_sha256:
        raise IdentityMismatch(
            f'checkpoint_sha256 mismatch: expected {expect_checkpoint_sha256}, '
            f'got {report["checkpoint_sha256"]} -- the checkpoint file itself changed')
    if expect_network_signature is not None and report['network_signature'] != expect_network_signature:
        raise IdentityMismatch(
            f'network_signature mismatch: expected {expect_network_signature}, '
            f'got {report["network_signature"]} -- state_dict loaded without error '
            f'(same keys/shapes) but this network computes a DIFFERENT function than '
            f'the one pinned as expected. This is exactly the shape-compatible-wrong-'
            f'decoder case load_state_dict alone cannot catch.')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--output', required=True)
    p.add_argument('--expect-checkpoint-sha256', default=None)
    p.add_argument('--expect-network-signature', default=None)
    args = p.parse_args()
    try:
        report = verify(args.checkpoint, args.expect_checkpoint_sha256, args.expect_network_signature)
    except IdentityMismatch as exc:
        print(f'REJECT: {exc}', file=sys.stderr)
        raise SystemExit(1)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['summary']))
