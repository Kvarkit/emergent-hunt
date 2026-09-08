import tempfile
import unittest
from pathlib import Path

import torch

from emergent_hunt.diagnose_checkpoint import diagnose
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


if __name__ == '__main__':
    unittest.main()
