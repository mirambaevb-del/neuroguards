"""Unit tests for the conservative research-only preictal labeler."""
import unittest

from prediction_research import PredictionProtocol, candidate_preictal_label


class CandidatePreictalLabelTests(unittest.TestCase):
    def test_positive_does_not_need_30_minutes_after_seizure_onset(self):
        # Seizure begins at 1800s; the recording ends at 1900s.
        # The 300-304s window is wholly within the 30-to-10-minute interval.
        # It is valid positive despite not observing 30 minutes after onset.
        result = candidate_preictal_label(
            300, 304, [(1800, 1860)],
            continuous_intervals=[(0, 1900)],
            annotation_verified_through=1900,
        )
        self.assertEqual(result, 1)

    def test_negative_requires_full_future_horizon(self):
        result = candidate_preictal_label(
            100, 104, [],
            continuous_intervals=[(0, 2000)],
            annotation_verified_through=2000,
        )
        self.assertEqual(result, 0)

    def test_negative_is_unknown_without_full_horizon(self):
        result = candidate_preictal_label(
            100, 104, [],
            continuous_intervals=[(0, 1000)],
            annotation_verified_through=1000,
        )
        self.assertIsNone(result)

    def test_seizure_too_soon_is_ambiguous_not_negative(self):
        # Event is 5 minutes after the window, inside the minimum warning zone.
        result = candidate_preictal_label(
            100, 104, [(404, 450)],
            continuous_intervals=[(0, 2000)],
            annotation_verified_through=2000,
        )
        self.assertIsNone(result)

    def test_postictal_window_is_excluded(self):
        result = candidate_preictal_label(
            2000, 2004, [(1000, 1500)],
            continuous_intervals=[(0, 5000)],
            annotation_verified_through=5000,
        )
        self.assertIsNone(result)

    def test_window_crossing_preictal_boundary_is_excluded(self):
        # For onset=1800, the preictal interval is [0, 1200].
        result = candidate_preictal_label(
            1198, 1202, [(1800, 1860)],
            continuous_intervals=[(0, 5000)],
            annotation_verified_through=5000,
        )
        self.assertIsNone(result)

    def test_recording_gap_is_not_bridged(self):
        result = candidate_preictal_label(
            100, 104, [],
            continuous_intervals=[(0, 500), (600, 2000)],
            annotation_verified_through=2000,
        )
        self.assertIsNone(result)

    def test_invalid_protocol_is_rejected(self):
        with self.assertRaises(ValueError):
            PredictionProtocol(postictal_exclusion_seconds=-1)

    def test_invalid_seizure_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            candidate_preictal_label(
                100, 104, [(500, 500)],
                continuous_intervals=[(0, 2000)],
                annotation_verified_through=2000,
            )


if __name__ == "__main__":
    unittest.main()
