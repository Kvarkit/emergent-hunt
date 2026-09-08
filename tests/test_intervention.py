import unittest
from emergent_hunt.intervention import build_rows, control_summary, counterfactual_pairs


class InterventionTests(unittest.TestCase):
    def test_162_directed_pairs_54_per_factor(self):
        pairs = counterfactual_pairs()
        self.assertEqual(len(pairs), 162)
        by_factor = {'p': 0, 'd': 0, 't': 0}
        for base, cf, factor in pairs:
            by_factor[factor] += 1
            self.assertNotEqual(base, cf)
            # exactly one coordinate differs
            diffs = sum(1 for a, b in zip((base.prey, base.direction, base.trap),
                                           (cf.prey, cf.direction, cf.trap)) if a != b)
            self.assertEqual(diffs, 1)
        self.assertEqual(by_factor, {'p': 54, 'd': 54, 't': 54})

    def test_manual_control_tallies_match_25120(self):
        rows = build_rows()
        self.assertEqual(len(rows), 486)
        summary = control_summary(rows)

        for policy in ('handwritten', 'holistic'):
            self.assertEqual(summary[policy]['correct_before'], 162)  # 27/27, x6 reps
            self.assertEqual(summary[policy]['changed_by_factor'], {'p': 54, 'd': 54, 't': 0})

        self.assertEqual(summary['no_message']['correct_before'], 18)  # 3/27, x6 reps
        self.assertEqual(summary['no_message']['changed_by_factor'], {'p': 0, 'd': 0, 't': 0})

    def test_base_and_counterfactual_split_labels_present(self):
        rows = build_rows()
        for row in rows:
            self.assertIn(row['base_split'], ('train', 'test'))
            self.assertIn(row['counterfactual_split'], ('train', 'test'))

    def test_summary_after_matches_direct_row_aggregation(self):
        # regression for the n_after/correct_after aggregation bug (melioralab-agent #25222)
        rows = build_rows()
        summary = control_summary(rows)
        for policy in ('handwritten', 'holistic', 'no_message'):
            policy_rows = [r for r in rows if r['policy'] == policy]
            self.assertEqual(summary[policy]['n_after'], len(policy_rows))
            self.assertEqual(summary[policy]['correct_after'],
                              sum(int(r['correct_after']) for r in policy_rows))
        self.assertEqual(summary['handwritten']['correct_after'], 162)
        self.assertEqual(summary['holistic']['correct_after'], 162)
        self.assertEqual(summary['no_message']['correct_after'], 18)


if __name__ == '__main__':
    unittest.main()
