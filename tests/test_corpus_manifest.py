import unittest

from emergent_hunt.environment import corpus_manifest_sha256

# nadir-codex #25734 gate 3, second half: pins the actual n=3/by='pair' grid
# used to measure the topsim numbers in test_topsim.py, independent of the
# checkpoint_sha256 pin there. A checkpoint hash alone can't distinguish "the
# model changed" from "the data grid a metric was computed against changed"
# (a reordering, a held-out boundary edit, an off-by-one) -- this manifest is
# the other provenance half that has to match before comparing metrics across
# runs.
_EXPECTED_ALL_TRIPLE = '00ab64692dfaf7edab9d8467f95345f2e753f8ee02b89d71c60f1faa27c6687c'


class CorpusManifestTests(unittest.TestCase):
    def test_deterministic_and_stable_across_calls(self):
        self.assertEqual(corpus_manifest_sha256(3, 'all', 'triple'),
                          corpus_manifest_sha256(3, 'all', 'triple'))

    def test_split_and_by_and_n_all_change_the_manifest(self):
        base = corpus_manifest_sha256(3, 'all', 'triple')
        self.assertNotEqual(base, corpus_manifest_sha256(3, 'train', 'triple'))
        self.assertNotEqual(base, corpus_manifest_sha256(3, 'all', 'pair'))
        self.assertNotEqual(base, corpus_manifest_sha256(4, 'all', 'triple'))

    def test_pinned_against_the_grid_topsim_was_measured_on(self):
        # If this ever fails, either states()/held_out_pairs() changed (a
        # real data-grid drift a metric comparison needs to know about), or
        # this constant needs re-pinning against a deliberate, reviewed
        # change -- not silently bumped to whatever the new value is.
        self.assertEqual(corpus_manifest_sha256(3, 'all', 'triple'), _EXPECTED_ALL_TRIPLE)


if __name__ == '__main__':
    unittest.main()
