"""Frozen greedy-policy EH-INT-r0.1 export. No optimization or RNG sampling."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from .train import SlotSender, SlotReceiver, mlp, features
from .intervention import build_rows, control_summary


def diagnose(path):
    payload = Path(path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    saved = torch.load(path, map_location='cpu', weights_only=True)
    head = saved['head']
    sender = SlotSender() if saved['architecture'] == 'slots' else mlp(6, 16)
    receiver = SlotReceiver() if saved['architecture'] == 'slots' else mlp(19, 6 if head == 'factorized' else 9)
    sender.load_state_dict(saved['sender'])
    receiver.load_state_dict(saved['receiver'])
    sender.eval()
    receiver.eval()

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
    return {'checkpoint_sha256':digest, 'summary':control_summary(rows), 'rows':rows}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    report = diagnose(args.checkpoint)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['summary']))
