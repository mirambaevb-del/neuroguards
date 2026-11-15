"""Regression tests for partial-dataset training readiness."""
import unittest

from train import validate_training_audit


class PartialDatasetTrainingGateTests(unittest.TestCase):
    def test_verified_subset_can_train_when_other_edfs_are_rejected(self):
        audit = {
            "status": "audited",
            "accepted_recordings": [{"file": "chb01_01.edf"}],
            "rejected_recordings": [
                {"file": "chb01_02.edf", "reason": "Patient summary missing"},
                {"file": "chb01_03.edf", "reason": "File absent from summary"},
            ],
        }
        self.assertEqual(validate_training_audit(audit), (1, 2))

    def test_no_accepted_recordings_blocks_training(self):
        audit = {
            "status": "blocked_no_valid_recordings",
            "accepted_recordings": [],
            "rejected_recordings": [{"file": "bad.edf", "reason": "invalid"}],
        }
        with self.assertRaises(SystemExit):
            validate_training_audit(audit)

    def test_missing_dataset_blocks_training(self):
        audit = {
            "status": "blocked_missing_dataset",
            "accepted_recordings": [],
            "rejected_recordings": [],
        }
        with self.assertRaises(SystemExit):
            validate_training_audit(audit)

    def test_rejected_recordings_must_be_reported_as_a_list(self):
        audit = {
            "status": "audited",
            "accepted_recordings": [{"file": "chb01_01.edf"}],
            "rejected_recordings": None,
        }
        with self.assertRaises(SystemExit):
            validate_training_audit(audit)


if __name__ == "__main__":
    unittest.main()
