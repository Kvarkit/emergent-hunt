import json
import tempfile
import unittest
from pathlib import Path

from emergent_hunt.dev_cycles import cycle_config, run_cycles


class DevelopmentCycleTests(unittest.TestCase):
    def test_configuration_schedule_repeats_with_new_seed(self):
        self.assertEqual(cycle_config(0)[:4], (2, False, 16, 'one_way'))
        self.assertEqual(cycle_config(16)[-1], 1)

    def test_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / 'cycles.json')
            first = run_cycles(total=1, episodes=2, output=output)
            second = run_cycles(total=2, episodes=2, output=output)
            self.assertEqual(len(first), 1)
            self.assertEqual(len(second), 2)
            self.assertEqual(json.loads(Path(output).read_text())[-1]['cycle'], 2)


if __name__ == '__main__':
    unittest.main()
