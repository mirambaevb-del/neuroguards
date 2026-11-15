"""Regression tests for participant-aware CHB-MIT data splitting."""
import unittest

from neuroguard_core import split_records_three_way, subject_id_for_case


def records(*case_ids):
    return [{"patient": case_id, "file": f"{case_id}_01.edf"} for case_id in case_ids]


class ParticipantSplitTests(unittest.TestCase):
    def test_chb01_and_chb21_are_same_subject(self):
        self.assertEqual(subject_id_for_case("chb01"), subject_id_for_case("chb21"))

    def test_same_subject_cannot_cross_train_and_test(self):
        data = records("chb01", "chb02", "chb03", "chb21")
        split = {
            "train": ["chb01", "chb02"],
            "validation": ["chb03"],
            "test": ["chb21"],
        }
        with self.assertRaisesRegex(ValueError, "Participant leakage"):
            split_records_three_way(data, split)

    def test_related_cases_may_stay_in_same_partition(self):
        data = records("chb01", "chb02", "chb03", "chb21", "chb04")
        split = {
            "train": ["chb01", "chb21", "chb02"],
            "validation": ["chb03"],
            "test": ["chb04"],
        }
        train, validation, test = split_records_three_way(data, split)
        self.assertEqual({r["patient"] for r in train}, {"chb01", "chb21", "chb02"})
        self.assertEqual({r["patient"] for r in validation}, {"chb03"})
        self.assertEqual({r["patient"] for r in test}, {"chb04"})

    def test_empty_explicit_split_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly train"):
            split_records_three_way(
                records("chb01", "chb02", "chb03"), {}
            )

    def test_missing_or_extra_case_is_rejected(self):
        data = records("chb01", "chb02", "chb03", "chb04")
        split = {
            "train": ["chb01"],
            "validation": ["chb02"],
            "test": ["chb03"],
        }
        with self.assertRaisesRegex(ValueError, "cover exactly"):
            split_records_three_way(data, split)

    def test_duplicate_case_is_rejected(self):
        data = records("chb01", "chb02", "chb03")
        split = {
            "train": ["chb01", "chb01"],
            "validation": ["chb02"],
            "test": ["chb03"],
        }
        with self.assertRaisesRegex(ValueError, "duplicate"):
            split_records_three_way(data, split)


if __name__ == "__main__":
    unittest.main()
