"""Unit tests for event-level and false-alarm evaluation metrics."""
import unittest

import numpy as np

from neuroguard_core import Config
from train import event_metrics


class EventMetricsTests(unittest.TestCase):
    def setUp(self):
        self.config = Config()

    def test_rejects_misaligned_metadata_and_arrays(self):
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            event_metrics([], [{"file": "a.edf"}], np.array([0, 0]),
                          np.array([0, 0]), self.config)

    def test_rejects_non_binary_labels(self):
        with self.assertRaisesRegex(ValueError, "binary labels"):
            event_metrics([], [{"file": "a.edf"}], np.array([0, 2]),
                          np.array([0, 1]), self.config)

    def test_false_alarm_runs_split_at_time_gaps_and_file_boundaries(self):
        metadata = [
            {"file": "a.edf", "start_seconds": 0, "end_seconds": 4},
            {"file": "a.edf", "start_seconds": 4, "end_seconds": 8},
            {"file": "a.edf", "start_seconds": 12, "end_seconds": 16},
            {"file": "b.edf", "start_seconds": 0, "end_seconds": 4},
        ]
        y = np.array([0, 0, 0, 0])
        prediction = np.array([1, 1, 1, 1])
        result = event_metrics([], metadata, y, prediction, self.config)
        self.assertEqual(result["false_alarm_runs"], 3)
        self.assertAlmostEqual(result["interictal_exposure_hours"], 16 / 3600)
        self.assertAlmostEqual(result["false_alarms_per_interictal_hour"], 3 / (16 / 3600))

    def test_false_alarm_runs_are_order_independent_and_use_window_end(self):
        metadata = [
            {"file": "a.edf", "start_seconds": 4, "end_seconds": 8},
            {"file": "a.edf", "start_seconds": 0, "end_seconds": 4},
            {"file": "a.edf", "start_seconds": 12, "end_seconds": 16},
        ]
        # All are positive interictal windows. The first two are contiguous
        # despite their input order; the third starts a new run after a gap.
        result = event_metrics([], metadata, np.array([0, 0, 0]),
                               np.array([1, 1, 1]), self.config)
        self.assertEqual(result["false_alarm_runs"], 2)

    def test_nonpositive_window_separates_two_false_alarm_runs(self):
        metadata = [
            {"file": "a.edf", "start_seconds": 0, "end_seconds": 4},
            {"file": "a.edf", "start_seconds": 4, "end_seconds": 8},
            {"file": "a.edf", "start_seconds": 8, "end_seconds": 12},
        ]
        result = event_metrics([], metadata, np.array([0, 0, 0]),
                               np.array([1, 0, 1]), self.config)
        self.assertEqual(result["false_alarm_runs"], 2)

    def test_event_detection_uses_first_positive_fully_ictal_window(self):
        records = [{"file": "a.edf", "seizures": [(8, 20)]}]
        metadata = [
            {"file": "a.edf", "start_seconds": 8, "end_seconds": 12},
            {"file": "a.edf", "start_seconds": 12, "end_seconds": 16},
            {"file": "a.edf", "start_seconds": 16, "end_seconds": 20},
        ]
        result = event_metrics(records, metadata, np.array([1, 1, 1]),
                               np.array([0, 1, 0]), self.config)
        self.assertEqual(result["scorable_events"], 1)
        self.assertEqual(result["detected_events"], 1)
        self.assertEqual(result["latency_seconds_from_onset_to_window_end"], [8])

    def test_event_without_fully_ictal_window_is_unscorable(self):
        records = [{"file": "a.edf", "seizures": [(9, 11)]}]
        metadata = [{"file": "a.edf", "start_seconds": 0, "end_seconds": 4}]
        result = event_metrics(records, metadata, np.array([0]),
                               np.array([0]), self.config)
        self.assertEqual(result["unscorable_events"], 1)
        self.assertEqual(result["event_sensitivity_scorable"], None)
        self.assertEqual(result["event_sensitivity_all_annotated"], 0)


if __name__ == "__main__":
    unittest.main()
