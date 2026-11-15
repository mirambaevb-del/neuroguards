"""Regression tests for event-level scorable and unscorable counts."""
import unittest

import numpy as np

from neuroguard_core import Config
from train import event_metrics


class EventMetricTests(unittest.TestCase):
    def test_short_event_without_fully_ictal_window_is_unscorable(self):
        records = [{
            "file": "chb01_01.edf",
            "seizures": [(5.0, 9.0), (10.0, 18.0)],
        }]
        # No 4-second window is fully inside the 5-9 second event.
        # The 10-18 second event has two fully-ictal windows.
        metadata = [
            {"file": "chb01_01.edf", "start_seconds": 10.0, "end_seconds": 14.0},
            {"file": "chb01_01.edf", "start_seconds": 14.0, "end_seconds": 18.0},
        ]
        y = np.array([1, 1])
        prediction = np.array([1, 0])

        result = event_metrics(records, metadata, y, prediction, Config())

        self.assertEqual(result["annotated_events"], 2)
        self.assertEqual(result["scorable_events"], 1)
        self.assertEqual(result["unscorable_events"], 1)
        self.assertEqual(result["detected_events"], 1)
        self.assertEqual(result["missed_scorable_events"], 0)
        self.assertEqual(result["event_sensitivity_scorable"], 1.0)
        self.assertEqual(result["event_sensitivity_all_annotated"], 0.5)


if __name__ == "__main__":
    unittest.main()
