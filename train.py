"""Auditable, patient-independent CHB-MIT NeuroGuard baseline training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from neuroguard_core import (Config, VERSION, discover_recordings, feature_names,
                             iter_record_windows, split_records_three_way, subject_id_for_case)


def write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, allow_nan=False), encoding="utf-8")


def load_feature_matrix(records: list[dict], config: Config):
    features, labels, metadata, rejected = [], [], [], []
    for record in records:
        for feat, target, info in iter_record_windows(record, config, rejected):
            features.append(feat)
            labels.append(target)
            metadata.append(info)
    if not features:
        raise ValueError("No usable labeled EEG windows")
    return np.stack(features), np.asarray(labels, dtype=int), metadata, rejected


def event_metrics(records: list[dict], metadata: list[dict], y: np.ndarray,
                  prediction: np.ndarray, config: Config) -> dict:
    """An event is detected if a fully-ictal window is predicted positive.

    False alarm = the START of a consecutive positive run in fully interictal
    windows. Exposure denominator excludes ictal and boundary windows.
    """
    y = np.asarray(y)
    prediction = np.asarray(prediction)
    if y.ndim != 1 or prediction.ndim != 1:
        raise ValueError("y and prediction must be one-dimensional")
    if len(metadata) != len(y) or len(prediction) != len(y):
        raise ValueError("metadata, y, and prediction must have equal lengths")
    if not np.isin(y, [0, 1]).all() or not np.isin(prediction, [0, 1]).all():
        raise ValueError("y and prediction must contain only binary labels 0 and 1")

    all_events = [(r["file"], onset, offset) for r in records for onset, offset in r["seizures"]]
    latencies, hits, scorable_events = [], 0, 0
    for filename, onset, offset in all_events:
        fully_ictal_windows = [
            (m, p) for m, p in zip(metadata, prediction)
            if m["file"] == filename and
            m["start_seconds"] >= onset and m["end_seconds"] <= offset
        ]
        if not fully_ictal_windows:
            continue
        scorable_events += 1
        positive_ends = [m["end_seconds"] for m, p in fully_ictal_windows if p == 1]
        if positive_ends:
            hits += 1
            latencies.append(min(positive_ends) - onset)
    unscorable_events = len(all_events) - scorable_events
    ordered = sorted(
        zip(metadata, y, prediction),
        key=lambda item: (item[0]["file"], item[0]["start_seconds"]),
    )
    for m, _, _ in ordered:
        start, end = float(m["start_seconds"]), float(m["end_seconds"])
        if not np.isfinite(start) or not np.isfinite(end) or start < 0 or end <= start:
            raise ValueError("metadata must contain finite, valid window timestamps")

    false_alarms = 0
    previous_file, previous_end, previous_positive = None, None, False
    for m, target, pred in ordered:
        if target == 0 and pred == 1:
            contiguous = (previous_positive and previous_file == m["file"] and
                          previous_end is not None and
                          abs(m["start_seconds"] - previous_end) < 1e-9)
            if not contiguous:
                false_alarms += 1
        previous_positive = (target == 0 and pred == 1)
        previous_file, previous_end = m["file"], m["end_seconds"]
    # iter_record_windows emits non-overlapping, fixed-duration windows. Thus
    # each retained interictal window contributes exactly one window duration;
    # excluded/invalid windows do not inflate the exposure denominator.
    exposure_h = np.sum(y == 0) * config.window_seconds / 3600.0
    return {
        "annotated_events": len(all_events),
        "scorable_events": scorable_events,
        "unscorable_events": unscorable_events,
        "detected_events": hits,
        "missed_scorable_events": scorable_events - hits,
        "missed_events": scorable_events - hits,
        "event_sensitivity_all_annotated": hits / len(all_events) if all_events else None,
        "event_sensitivity_scorable": hits / scorable_events if scorable_events else None,
        "event_sensitivity": hits / scorable_events if scorable_events else None,
        "latency_seconds_from_onset_to_window_end": latencies,
        "median_detection_latency_seconds": float(np.median(latencies)) if latencies else None,
        "false_alarm_runs": false_alarms,
        "interictal_exposure_hours": float(exposure_h),
        "false_alarms_per_interictal_hour": false_alarms / exposure_h if exposure_h else None,
        "event_definition": "One event detected if any fully-ictal 4 s window was predicted positive",
        "false_alarm_definition": "Start of each consecutive predicted-positive run among fully interictal windows",
    }


def evaluate(model, x: np.ndarray, y: np.ndarray, metadata: list[dict],
             records: list[dict], config: Config) -> tuple[dict, np.ndarray, np.ndarray]:
    pred = model.predict(x)
    scores = model.predict_proba(x)[:, list(model.classes_).index(1)]
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = map(int, matrix.ravel())
    result = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": tn / (tn + fp) if (tn + fp) else None,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "confusion_matrix_0_1": matrix.tolist(),
        "true_negatives": tn, "false_positives": fp, "false_negatives": fn, "true_positives": tp,
        "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
        "pr_auc_average_precision": float(average_precision_score(y, scores)) if len(np.unique(y)) == 2 else None,
        "held_out_cases": sorted({m["patient"] for m in metadata}),
        "held_out_subjects": sorted({subject_id_for_case(m["patient"]) for m in metadata}),
        "n_test_windows": len(y),
        "scores_are_calibrated_probabilities": False,
        "event_metrics": event_metrics(records, metadata, y, pred, config),
    }
    return result, pred, scores


def save_plots(results: Path, model, x: np.ndarray, y: np.ndarray, scores: np.ndarray,
               comparison: dict, first_record: dict | None = None,
               held_out_patients: list[str] | None = None):
    disp = ConfusionMatrixDisplay.from_predictions(y, model.predict(x), labels=[0, 1],
                                                   display_labels=["Interictal", "Ictal"], cmap="Blues")
    disp.figure_.savefig(results / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(disp.figure_)
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, scores)
        plt.figure()
        patient_label = ", ".join(held_out_patients or []) or "unspecified patients"
        plt.plot(fpr, tpr, label=f"Held-out patients: {patient_label}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
        plt.legend(); plt.tight_layout()
        plt.savefig(results / "roc_curve.png", dpi=150); plt.close()
        precision, recall, _ = precision_recall_curve(y, scores)
        plt.figure(); plt.plot(recall, precision)
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.tight_layout()
        plt.savefig(results / "precision_recall_curve.png", dpi=150); plt.close()
    plt.figure()
    plt.bar(list(comparison.keys()), [v["macro_f1"] for v in comparison.values()])
    plt.ylabel("Validation macro F1"); plt.ylim(0, 1); plt.tight_layout()
    plt.savefig(results / "model_comparison.png", dpi=150); plt.close()
    if first_record is not None:
        from neuroguard_core import load_edf, select_channels
        raw = load_edf(first_record["file"])
        try:
            names = select_channels(raw.ch_names)
            start = max(0, int(first_record["seizures"][0][0] - 20)) if first_record["seizures"] else 0
            stop = min(raw.n_times, int((start + 90) * raw.info["sfreq"]))
            sample = raw.get_data(picks=[names[0]], start=int(start * raw.info["sfreq"]), stop=stop)[0]
            ts = np.arange(len(sample)) / raw.info["sfreq"] + start
            plt.figure(figsize=(12, 3)); plt.plot(ts, sample * 1e6, linewidth=.65)
            for a, b in first_record["seizures"]:
                if a <= ts[-1] and b >= ts[0]:
                    plt.axvspan(a, b, color="tomato", alpha=.25)
            plt.xlabel("Seconds from recording start"); plt.ylabel("EEG, microvolts")
            plt.title(f"{Path(first_record['file']).name}: {names[0]} (shaded = annotated seizure)")
            plt.tight_layout(); plt.savefig(results / "eeg_with_annotations.png", dpi=150); plt.close()
        finally:
            raw.close()


def main():
    parser = argparse.ArgumentParser(description="NeuroGuard patient-independent research baseline")
    parser.add_argument("--data", default="/content/neuroguard_data")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--train", action="store_true", help="Fit models only if real labeled EDF data exists")
    parser.add_argument("--split-json", help="Optional JSON file with disjoint train/validation/test patient lists")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    results, model_dir = out / "results", out / "model"
    results.mkdir(parents=True, exist_ok=True)
    config = Config()
    audit = discover_recordings(args.data, config)
    write_json(results / "dataset_audit.json", audit)
    print(f"Dataset audit: {audit['status']}; files={audit['edf_files_found']}; accepted={len(audit['accepted_recordings'])}")
    if not args.train:
        print("Audit-only. Add --train to train on actual EEG recordings.")
        return
    if audit["status"] != "audited" or audit["rejected_recordings"]:
        raise SystemExit("Cannot train: missing/invalid data or rejected recordings. Review results/dataset_audit.json.")
    split_patients = None
    if args.split_json:
        with Path(args.split_json).open("r", encoding="utf-8") as handle:
            split_patients = json.load(handle)
        if (not isinstance(split_patients, dict) or
                set(split_patients) != {"train", "validation", "test"}):
            raise SystemExit("--split-json must define exactly train, validation and test lists")
    train_records, val_records, test_records = split_records_three_way(
        audit["accepted_recordings"], split_patients=split_patients)
    split_ids = {"train": sorted({r["patient"] for r in train_records}),
                 "validation": sorted({r["patient"] for r in val_records}),
                 "test": sorted({r["patient"] for r in test_records})}
    write_json(results / "patient_split.json", {"patients": split_ids,
                                                "method": "User-supplied participant-aware case split" if split_patients is not None else "Deterministic participant-aware proposal chosen after EDF audit",
                                                "warning": "Review suitability; only 3 patients gives very weak external validation"})
    x_train, y_train, metadata_train, rejected_train = load_feature_matrix(train_records, config)
    x_val, y_val, metadata_val, rejected_val = load_feature_matrix(val_records, config)
    x_test, y_test, metadata_test, rejected_test = load_feature_matrix(test_records, config)
    write_json(results / "rejected_windows.json", {"train": rejected_train,
                                                       "validation": rejected_val, "test": rejected_test})
    if len(set(y_train.tolist())) < 2:
        raise SystemExit("Cannot train a binary detector: training patients lack one of the two verified classes")
    if len(set(y_val.tolist())) < 2:
        raise SystemExit("Cannot tune a seizure detector: validation patients lack both classes")
    # Small, prespecified search; ONLY validation patients are used for selection.
    candidates = {
        "random_forest_leaf1": RandomForestClassifier(n_estimators=150, min_samples_leaf=1,
                                                       class_weight="balanced_subsample", random_state=42, n_jobs=-1),
        "random_forest_leaf2": RandomForestClassifier(n_estimators=150, min_samples_leaf=2,
                                                       class_weight="balanced_subsample", random_state=42, n_jobs=-1),
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ]),
    }
    validation_comparison = {}
    for name, clf in candidates.items():
        clf.fit(x_train, y_train)
        metric, _, _ = evaluate(clf, x_val, y_val, metadata_val, val_records, config)
        validation_comparison[name] = metric
        print(f"{name}: validation macro F1={metric['macro_f1']:.3f}")
    best_name = max(validation_comparison, key=lambda name: validation_comparison[name]["macro_f1"])
    # Refit on the combined train + validation PATIENTS; test is unseen.
    chosen = clone(candidates[best_name]).fit(np.concatenate([x_train, x_val]),
                                               np.concatenate([y_train, y_val]))
    test_metrics, _, chosen_scores = evaluate(chosen, x_test, y_test, metadata_test, test_records, config)
    patient_level = {}
    for patient in split_ids["test"]:
        indexes = [i for i, item in enumerate(metadata_test) if item["patient"] == patient]
        patient_records = [r for r in test_records if r["patient"] == patient]
        patient_level[patient], _, _ = evaluate(chosen, x_test[indexes], y_test[indexes],
                                                [metadata_test[i] for i in indexes], patient_records, config)
    test_metrics["by_patient"] = patient_level
    print(f"Selected on validation: {best_name}; TEST macro F1={test_metrics['macro_f1']:.3f}")
    model_dir.mkdir(exist_ok=True, parents=True)
    joblib.dump(chosen, model_dir / "seizure_detector.joblib")
    model_hash = hashlib.sha256((model_dir / "seizure_detector.joblib").read_bytes()).hexdigest()
    write_json(model_dir / "config.json", config.to_json())
    write_json(model_dir / "features.json", {"feature_names": feature_names(config), "version": VERSION,
                                                "class_labels": {"0": "INTERICTAL", "1": "ICTAL"}})
    write_json(model_dir / "manifest.json", {"version": VERSION, "trained_utc": datetime.now(timezone.utc).isoformat(),
                                                "training_cases": split_ids["train"] + split_ids["validation"],
                                                "held_out_cases": split_ids["test"], "validation_cases": split_ids["validation"],
                                                "training_subjects": sorted({subject_id_for_case(r["patient"]) for r in train_records + val_records}),
                                                "held_out_subjects": sorted({subject_id_for_case(r["patient"]) for r in test_records}),
                                                "selected_model": best_name, "model_sha256": model_hash,
                                                "selection": "Choose best validation macro F1, refit train+val, evaluate test once",
                                                "python_version": platform.python_version(),
                                                "libraries": {name: importlib.metadata.version(name) for name in
                                                              ["numpy", "scipy", "scikit-learn", "mne", "joblib"]},
                                                "calibration": "none", "data_source": "user-supplied CHB-MIT EDF and summary files"})
    write_json(results / "validation_comparison.json", validation_comparison)
    write_json(results / "test_metrics.json", test_metrics)
    save_plots(results, chosen, x_test, y_test, chosen_scores, validation_comparison,
               next((r for r in test_records if r["seizures"]), None),
               held_out_patients=split_ids["test"])
    print(f"Saved trained model and actual held-out metrics to {out}")


if __name__ == "__main__":
    main()
