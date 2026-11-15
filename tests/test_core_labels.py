"""Regression tests for ictal, interictal, and postictal window labels."""
import unittest

from neuroguard_core import (
    POSTICTAL_EXCLUSION_SECONDS,
    label_window,
    window_exclusion_reason,
)


class LabelWindowTests(unittest.TestCase):
    def test_fully_ictal_window_is_positive(self):
        self.assertEqual(label_window(100, 104, [(90, 120)]), 1)

    def test_window_before_seizure_is_interictal(self):
        self.assertEqual(label_window(50, 54, [(100, 120)]), 0)

    def test_window_overlapping_seizure_boundary_is_unknown(self):
        self.assertIsNone(label_window(88, 92, [(90, 120)]))

    def test_window_at_seizure_end_is_postictal_unknown(self):
        self.assertIsNone(label_window(120, 124, [(90, 120)]))

    def test_window_inside_postictal_exclusion_is_unknown(self):
        self.assertIsNone(label_window(600, 604, [(90, 120)]))

    def test_window_partly_overlapping_postictal_exclusion_is_unknown(self):
        end_of_exclusion = 120 + POSTICTAL_EXCLUSION_SECONDS
        self.assertIsNone(label_window(end_of_exclusion - 2, end_of_exclusion + 2, [(90, 120)]))

    def test_window_at_exact_end_of_postictal_exclusion_is_interictal(self):
        end_of_exclusion = 120 + POSTICTAL_EXCLUSION_SECONDS
        self.assertEqual(label_window(end_of_exclusion, end_of_exclusion + 4, [(90, 120)]), 0)

    def test_custom_postictal_duration_is_respected(self):
        self.assertIsNone(label_window(200, 204, [(90, 120)], postictal_exclusion_seconds=120))
        self.assertEqual(label_window(240, 244, [(90, 120)], postictal_exclusion_seconds=120), 0)

    def test_zero_postictal_duration_disables_postictal_exclusion(self):
        self.assertEqual(label_window(120, 124, [(90, 120)], postictal_exclusion_seconds=0), 0)

    def test_overlapping_or_unsorted_seizure_intervals_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "sorted and non-overlapping"):
            label_window(10, 14, [(10, 30), (25, 40)])


    def test_exclusion_reason_distinguishes_postictal_from_seizure_boundary(self):
        intervals = [(90, 120)]
        self.assertEqual(window_exclusion_reason(124, 128, intervals), "postictal_exclusion")
        self.assertEqual(window_exclusion_reason(88, 92, intervals), "seizure_boundary")
        self.assertIsNone(window_exclusion_reason(50, 54, intervals))

    def test_invalid_postictal_duration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            label_window(10, 14, [(30, 40)], postictal_exclusion_seconds=-1)


if __name__ == "__main__":
    unittest.main()
