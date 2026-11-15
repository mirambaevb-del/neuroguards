"""API tests with a deterministic fake detector; no trained model or EEG data required."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import api


class FakeDetector:
    def __init__(self, model_dir=None):
        self.manifest = {
            "version": "test-model",
            "training_cases": ["chb01", "chb02"],
            "training_subjects": ["subject_chb01_chb21", "chb02"],
            "held_out_cases": ["chb03"],
            "held_out_subjects": ["chb03"],
        }
        self.config = type("Config", (), {
            "channels": ("FP1-F7", "F7-T7", "T7-P7", "P7-O1",
                         "FP2-F8", "F8-T8", "T8-P8", "P8-O2"),
            "sfreq": 256.0, "window_seconds": 4.0,
        })()

    def predict_array(self, samples, channels, sfreq, start_seconds, *, unit):
        return {"status": "ok", "task": "SEIZURE_DETECTION_NOT_PREDICTION",
                "windows": [{"label": "INTERICTAL"}]}

    def predict_edf(self, path):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("missing EDF bytes")
        return {"status": "ok", "windows": []}


VALID_PAYLOAD = {
    "channels": ["FP1-F7", "F7-T7", "T7-P7", "P7-O1",
                 "FP2-F8", "F8-T8", "T8-P8", "P8-O2"],
    "sfreq": 256, "start_seconds": 0, "unit": "V",
    "samples": [[0.0, 0.0, 0.0, 0.0] for _ in range(8)],
}


class APITests(unittest.TestCase):
    def client(self, **env):
        values = {
            "NEUROGUARD_PUBLIC": "0",
            "NEUROGUARD_API_KEY": "",
            "NEUROGUARD_ALLOWED_ORIGINS": "",
            "NEUROGUARD_MODEL_DIR": "",
        }
        values.update(env)
        stack = patch.dict(os.environ, values, clear=False)
        stack.start()
        detector_patch = patch.object(api, "NeuroGuardDetector", FakeDetector)
        detector_patch.start()
        self.addCleanup(detector_patch.stop)
        self.addCleanup(stack.stop)
        client = TestClient(api.app)
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        return client

    def test_health_reports_loaded_model(self):
        response = self.client().get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["model_loaded"])
        self.assertFalse(response.json()["medical_device"])

    def test_model_info_reports_detection_not_prediction(self):
        response = self.client().get("/model-info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["task"], "binary_seizure_detection")
        self.assertEqual(response.json()["training_cases"], ["chb01", "chb02"])
        self.assertEqual(response.json()["training_subjects"], ["subject_chb01_chb21", "chb02"])
        self.assertEqual(response.json()["held_out_cases"], ["chb03"])
        self.assertEqual(response.json()["held_out_subjects"], ["chb03"])
        self.assertNotIn("training_patients", response.json())
        self.assertFalse(response.json()["clinical_validation"])
    
    def test_model_info_split_fields_match_training_manifest_schema(self):
        root = Path(__file__).resolve().parents[1]
        training_source = (root / "train.py").read_text(encoding="utf-8")
        api_source = (root / "api.py").read_text(encoding="utf-8")
        for field in ("training_cases", "training_subjects", "held_out_cases", "held_out_subjects"):
            with self.subTest(field=field):
                self.assertIn(f'"{field}"', training_source)
                self.assertIn(f'"{field}"', api_source)

    def test_predict_accepts_valid_eeg_payload(self):
        response = self.client().post("/predict", json=VALID_PAYLOAD)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["task"], "SEIZURE_DETECTION_NOT_PREDICTION")

    def test_predict_rejects_non_volt_unit(self):
        payload = {**VALID_PAYLOAD, "unit": "uV"}
        response = self.client().post("/predict", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_predict_rejects_channel_sample_row_mismatch(self):
        payload = {**VALID_PAYLOAD, "samples": VALID_PAYLOAD["samples"][:-1]}
        response = self.client().post("/predict", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_predict_enforces_json_body_limit(self):
        response = self.client().post(
            "/predict", content=b"x" * (4 * 1024 * 1024 + 1),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 413)

    def test_predict_edf_accepts_raw_nonempty_bytes(self):
        response = self.client().post(
            "/predict-edf", content=b"fake-edf-bytes",
            headers={"Content-Type": "application/octet-stream"},
        )
        self.assertEqual(response.status_code, 200)

    def test_configured_api_key_is_required(self):
        response = self.client(NEUROGUARD_API_KEY="secret").get("/model-info")
        self.assertEqual(response.status_code, 401)
        response = TestClient(api.app)
        with patch.dict(os.environ, {
            "NEUROGUARD_PUBLIC": "0", "NEUROGUARD_API_KEY": "secret",
        }, clear=False), patch.object(api, "NeuroGuardDetector", FakeDetector):
            with response:
                authorized = response.get("/model-info", headers={"X-API-Key": "secret"})
        self.assertEqual(authorized.status_code, 200)

    def test_public_mode_refuses_start_without_api_key(self):
        with patch.dict(os.environ, {
            "NEUROGUARD_PUBLIC": "1", "NEUROGUARD_API_KEY": "",
        }, clear=False):
            with patch.object(api, "NeuroGuardDetector", FakeDetector):
                with self.assertRaisesRegex(RuntimeError, "require NEUROGUARD_API_KEY"):
                    with TestClient(api.app):
                        pass


if __name__ == "__main__":
    unittest.main()
