import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from emergent_hunt.diagnose_checkpoint import diagnose, build_networks, fixed_probe_signature, verify, IdentityMismatch
from emergent_hunt.train import SlotSender, SlotReceiver, BagReceiver, mlp


def _save(tmpdir, architecture, head='factorized'):
    """Build an untrained network for `architecture` and save it in the
    exact format train.run() writes, so diagnose() sees a real checkpoint.
    """
    sender = SlotSender() if architecture == 'slots' else mlp(6, 16)
    if architecture == 'slots':
        receiver = SlotReceiver()
    elif architecture == 'receiver_slots':
        receiver = SlotReceiver()
    elif architecture == 'bag_receiver':
        receiver = BagReceiver()
    else:
        receiver = mlp(19, 6 if head == 'factorized' else 9)
    path = Path(tmpdir) / f'{architecture}.pt'
    torch.save({'sender': sender.state_dict(), 'receiver': receiver.state_dict(),
                'critic': mlp(9, 1).state_dict(), 'architecture': architecture, 'head': head,
                'seed': 0, 'mode': 'communication', 'by': 'pair', 'reward_kind': 'exact',
                'steps': 0}, path)
    return path


class DiagnoseCheckpointTests(unittest.TestCase):
    def test_round_trip_every_trained_architecture(self):
        # Regression for melioralab-agent #25638: bag_receiver checkpoints
        # fell through to the generic mlp(19, ...) branch and load_state_dict
        # raised on mismatched keys (net.0.weight vs 0.weight). Every
        # architecture train.run() can produce must diagnose cleanly.
        with tempfile.TemporaryDirectory() as tmp:
            for architecture in ('mlp', 'slots', 'receiver_slots', 'bag_receiver'):
                path = _save(tmp, architecture)
                report = diagnose(str(path))
                self.assertEqual(len(report['rows']), 162)
                self.assertEqual(report['rows'][0]['architecture'], architecture)

    def test_bag_receiver_state_dict_is_not_silently_accepted_by_generic_mlp(self):
        # Direct evidence the pre-fix dispatch was wrong, not just that the
        # new dispatch happens to work: the BagReceiver state_dict genuinely
        # does not fit the generic mlp(19, ...) shape used as the fallback.
        bag = BagReceiver()
        generic = mlp(19, 6)
        with self.assertRaises((RuntimeError, ValueError)):
            generic.load_state_dict(bag.state_dict())

    # --- nadir-codex #25734 gate 1: registry, not a fallback chain --------

    def test_unknown_architecture_tag_is_rejected(self):
        with self.assertRaises(ValueError):
            build_networks('typo_architecture', head='factorized')

    def test_every_train_cli_architecture_choice_is_registered(self):
        # train.py's --architecture choices are the ground truth for which
        # tags a checkpoint can legitimately carry; every one of them must
        # build without raising, or a valid checkpoint would be rejected.
        for architecture in ('mlp', 'slots', 'receiver_slots', 'bag_receiver'):
            build_networks(architecture, head='factorized')  # must not raise

    # --- nadir-codex #25734 gate 2: identity negative control -------------

    class _WrongOrderBagReceiver(nn.Module):
        """Same parameter names/shapes as BagReceiver (net = mlp(11, 6)), so
        the SAME state_dict loads into it without error -- but it feeds the
        trap one-hot and the pooled message bag to the linear layer in the
        opposite order, so a real BagReceiver's weights are wired to the
        wrong inputs here. This is the shape-compatible-but-wrong decoder
        nadir-codex #25649/#25734 asked for as a negative control."""
        def __init__(self):
            super().__init__()
            self.net = mlp(11, 6)

        def forward(self, x):
            return self.net(torch.cat((x[:, 3:].view(-1, 2, 8).sum(1), x[:, :3]), -1))

    def test_shape_compatible_wrong_decoder_is_caught_by_output_signature(self):
        # The failure mode gate 1 alone cannot catch: two receiver classes
        # can have identical state_dict keys and shapes (load_state_dict
        # raises on neither) while computing different functions. Only
        # comparing actual outputs on a fixed probe -- not "did loading
        # succeed" -- catches this.
        real = BagReceiver()
        wrong = self._WrongOrderBagReceiver()
        wrong.load_state_dict(real.state_dict())  # succeeds: same keys/shapes
        sender = mlp(6, 16)

        sig_real = fixed_probe_signature(sender, real, head='factorized')
        sig_wrong = fixed_probe_signature(sender, wrong, head='factorized')
        self.assertNotEqual(sig_real, sig_wrong,
                             'wrong-order decoder produced the same signature as the '
                             'real one -- fixed_probe_signature failed to catch a '
                             'shape-compatible identity mismatch')

    def test_round_trip_signature_is_stable_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _save(tmp, 'bag_receiver')
            report1 = diagnose(str(path))
            report2 = diagnose(str(path))
            self.assertEqual(report1['network_signature'], report2['network_signature'])

    # --- nadir-codex #25752: on-disk file, fresh subprocess -----------------

    def _run_diagnose_subprocess(self, checkpoint_path, output_path):
        # A genuinely fresh `python -m` process, not an in-process call: no
        # shared import state, no leftover module-level caching, nothing
        # this test file's own process happens to have already loaded.
        result = subprocess.run(
            [sys.executable, '-m', 'emergent_hunt.diagnose_checkpoint',
             str(checkpoint_path), '--output', str(output_path)],
            cwd=Path(__file__).parent.parent / 'src', capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(Path(output_path).read_text(encoding='utf-8'))

    def test_file_round_trip_in_fresh_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = _save(tmp, 'bag_receiver')
            output = Path(tmp) / 'report.json'
            report = self._run_diagnose_subprocess(checkpoint, output)

            expected_sha256 = __import__('hashlib').sha256(checkpoint.read_bytes()).hexdigest()
            self.assertEqual(report['checkpoint_sha256'], expected_sha256)
            self.assertEqual(report['rows'][0]['architecture'], 'bag_receiver')
            self.assertIn('network_signature', report)
            self.assertEqual(len(report['network_signature']), 64)  # sha256 hex

    def test_forged_on_disk_checkpoint_is_caught_by_output_signature(self):
        # Same request as gate 2, but end-to-end: a real .pt FILE on disk,
        # tagged bag_receiver, whose receiver state_dict was produced by the
        # wrong-order wiring -- strict load_state_dict inside diagnose() does
        # not raise (same keys/shapes), so only network_signature can tell
        # the two files apart. This is the file-level counterpart nadir-codex
        # #25752 asked for in addition to the in-memory gate-2 test above.
        with tempfile.TemporaryDirectory() as tmp:
            real_path = _save(tmp, 'bag_receiver')

            wrong = self._WrongOrderBagReceiver()
            wrong.load_state_dict(BagReceiver().state_dict())  # succeeds: same keys/shapes
            forged_path = Path(tmp) / 'forged_bag_receiver.pt'
            torch.save({'sender': mlp(6, 16).state_dict(), 'receiver': wrong.state_dict(),
                        'critic': mlp(9, 1).state_dict(), 'architecture': 'bag_receiver',
                        'head': 'factorized', 'seed': 0, 'mode': 'communication', 'by': 'pair',
                        'reward_kind': 'exact', 'steps': 0}, forged_path)

            real_report = self._run_diagnose_subprocess(real_path, Path(tmp) / 'real.json')
            forged_report = self._run_diagnose_subprocess(forged_path, Path(tmp) / 'forged.json')

            self.assertNotEqual(real_report['checkpoint_sha256'], forged_report['checkpoint_sha256'])
            self.assertNotEqual(
                real_report['network_signature'], forged_report['network_signature'],
                'forged on-disk checkpoint (wrong-order BagReceiver wiring, same '
                'state_dict shape) produced the same network_signature as the real '
                'one when loaded in a fresh subprocess -- strict state_dict load '
                'alone cannot catch this, and neither did the identity check')

    # --- nadir-codex #25800: a differing signature must REJECT, not just be
    # reported -------------------------------------------------------------

    def test_verify_accepts_matching_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _save(tmp, 'bag_receiver')
            baseline = diagnose(str(path))
            report = verify(str(path), baseline['checkpoint_sha256'], baseline['network_signature'])
            self.assertEqual(report['network_signature'], baseline['network_signature'])

    def test_verify_rejects_signature_mismatch(self):
        # nadir-codex #25800's exact falsifier: pin the expected signature of
        # the real checkpoint's architecture, then feed verify() a
        # shape-compatible forged decoder saved under the same tag. A
        # verifier that only checks "does diagnose() return a 64-hex
        # digest" would pass this; verify() must not.
        with tempfile.TemporaryDirectory() as tmp:
            real_path = _save(tmp, 'bag_receiver')
            expected = diagnose(str(real_path))['network_signature']

            wrong = self._WrongOrderBagReceiver()
            wrong.load_state_dict(BagReceiver().state_dict())
            forged_path = Path(tmp) / 'forged.pt'
            torch.save({'sender': mlp(6, 16).state_dict(), 'receiver': wrong.state_dict(),
                        'critic': mlp(9, 1).state_dict(), 'architecture': 'bag_receiver',
                        'head': 'factorized', 'seed': 0, 'mode': 'communication', 'by': 'pair',
                        'reward_kind': 'exact', 'steps': 0}, forged_path)

            with self.assertRaises(IdentityMismatch):
                verify(str(forged_path), expect_network_signature=expected)

    def test_verify_rejects_checkpoint_sha256_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _save(tmp, 'mlp')
            with self.assertRaises(IdentityMismatch):
                verify(str(path), expect_checkpoint_sha256='0' * 64)

    def test_cli_exits_nonzero_and_rejects_on_signature_mismatch(self):
        # The fail-closed path nadir-codex #25800 asked to see demonstrated:
        # a fresh subprocess, given --expect-network-signature for the real
        # checkpoint's architecture, must refuse (nonzero exit, no report
        # written) when pointed at the forged one -- not silently write a
        # report carrying a different-but-unenforced digest.
        with tempfile.TemporaryDirectory() as tmp:
            real_path = _save(tmp, 'bag_receiver')
            expected = diagnose(str(real_path))['network_signature']

            wrong = self._WrongOrderBagReceiver()
            wrong.load_state_dict(BagReceiver().state_dict())
            forged_path = Path(tmp) / 'forged.pt'
            torch.save({'sender': mlp(6, 16).state_dict(), 'receiver': wrong.state_dict(),
                        'critic': mlp(9, 1).state_dict(), 'architecture': 'bag_receiver',
                        'head': 'factorized', 'seed': 0, 'mode': 'communication', 'by': 'pair',
                        'reward_kind': 'exact', 'steps': 0}, forged_path)
            output = Path(tmp) / 'should_not_be_written.json'

            result = subprocess.run(
                [sys.executable, '-m', 'emergent_hunt.diagnose_checkpoint', str(forged_path),
                 '--output', str(output), '--expect-network-signature', expected],
                cwd=Path(__file__).parent.parent / 'src', capture_output=True, text=True)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn('REJECT', result.stderr)
            self.assertFalse(output.exists(),
                              'diagnose_checkpoint wrote a report for a checkpoint that '
                              'failed identity verification')

    def test_cli_exits_zero_when_signature_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _save(tmp, 'bag_receiver')
            expected = diagnose(str(path))['network_signature']
            output = Path(tmp) / 'report.json'
            result = subprocess.run(
                [sys.executable, '-m', 'emergent_hunt.diagnose_checkpoint', str(path),
                 '--output', str(output), '--expect-network-signature', expected],
                cwd=Path(__file__).parent.parent / 'src', capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.exists())


if __name__ == '__main__':
    unittest.main()
