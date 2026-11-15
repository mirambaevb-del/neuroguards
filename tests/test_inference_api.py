"""Tests for inference input validation and streaming packet continuity."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from inference import NeuroGuardDetector, StreamingEEGSession


CHANNELS = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1",
            "FP2-F8", "F8-T8", "T8-P8", "P8-O2"]


def fake_detector():
    detector = NeuroGuardDetector.__new__(NeuroGuardDetector)
    detector.config = SimpleNamespace(
        channels=tuple(CHANNELS), sfreq=256.0,
        samples_per_window=4, window_seconds=4.0,
    )
    detector.manifest = {"version": "test"}
    detector._predict_window = lambda window, start: {
        "label": "INTERICTAL", "start_seconds": start,
        "end_seconds": start + 4.0, "status": "ok",
    }
    return detector


class PredictArrayValidationTests(unittest.TestCase):
    def test_valid_volts_and_channel_order_produce_windows(self):
        detector = fake_detector()
        with patch("inference.select_channels", return_value=CHANNELS):
            result = detector.predict_array(
                np.zeros((8, 8)), CHANNELS, 256.0, unit="V"
            )
        self.assertEqual(result["task"], "SEIZURE_DETECTION_NOT_PREDICTION")
        self.assertEqual(len(result["windows"]), 2)

    def test_unit_must_be_volts(self):
        detector = fake_detector()
        with self.assertRaisesRegex(ValueError, "unit must be explicitly"):
            detector.predict_array(np.zeros((8, 4)), CHANNELS, 256.0, unit="uV")

    def test_sampling_frequency_must_match(self):
        detector = fake_detector()
        with self.assertRaisesRegex(ValueError, "Sampling frequency"):
            detector.predict_array(np.zeros((8, 4)), CHANNELS, 250.0, unit="V")

    def test_partial_window_is_rejected(self):
        detector = fake_detector()
        with patch("inference.select_channels", return_value=CHANNELS):
            with self.assertRaisesRegex(ValueError, "multiple"):
                detector.predict_array(np.zeros((8, 5)), CHANNELS, 256.0, unit="V")

    def test_non_finite_samples_are_rejected(self):
        detector = fake_detector()
        data = np.zeros((8, 4))
        data[0, 0] = np.nan
        with patch("inference.select_channels", return_value=CHANNELS):
            with self.assertRaisesRegex(ValueError, "NaN/Infinity"):
                detector.predict_array(data, CHANNELS, 256.0, unit="V")


class StreamingContinuityTests(unittest.TestCase):
    def setUp(self):
        self.detector = fake_detector()
        self.session = StreamingEEGSession(self.detector)

    def push(self, start, count=2):
        with patch("inference.select_channels", return_value=CHANNELS):
            return self.session.push(
                np.zeros((8, count)), CHANNELS, 256.0,
                unit="V", packet_start_seconds=start,
            )

    def test_fragmented_contiguous_packets_form_one_window(self):
        self.assertEqual(self.push(0.0), [])
        result = self.push(2 / 256)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["start_seconds"], 0.0)

    def test_packet_gap_is_rejected(self):
        self.push(0.0)
        with self.assertRaisesRegex(ValueError, "Non-contiguous EEG packet"):
            self.push(0.02)

    def test_overlapping_or_out_of_order_packet_is_rejected(self):
        self.push(0.0)
        with self.assertRaisesRegex(ValueError, "Non-contiguous EEG packet"):
            self.push(0.0)

    def test_empty_packet_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one sample"):
            self.push(0.0, count=0)

    def test_packet_timestamp_must_be_finite(self):
        with self.assertRaisesRegex(ValueError, "packet_start_seconds"):
            self.push(float("nan"))

    def test_session_rejects_packets_after_finalize(self):
        self.session.finalize()
        with self.assertRaisesRegex(ValueError, "already finalized"):
            self.push(0.0)


if __name__ == "__main__":
    unittest.main()
