"""NeuroGuard research-only EEG processing. No clinical use.

This module intentionally does not create synthetic training data, infer
missing seizure annotations, or assume an EEG recording is seizure-free.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import re

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch


VERSION = "0.2.0-research"
CHANNELS = (
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
)
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 40)}
FEATURES_PER_CHANNEL = (
    "mean", "std", "variance", "rms", "energy", "peak_to_peak",
    "zero_crossing_rate", "hjorth_mobility",
    "hjorth_complexity", "delta_power", "theta_power", "alpha_power",
    "beta_power", "gamma_power", "delta_relative", "theta_relative",
    "alpha_relative", "beta_relative", "gamma_relative", "spectral_entropy",
)


@dataclass(frozen=True)
class Config:
    sfreq: int = 256
    window_seconds: int = 4
    low_hz: float = 0.5
    high_hz: float = 40.0
    channels: tuple[str, ...] = CHANNELS
    notch_hz: float | None = None  # Disabled: no verified local mains artifact.

    @property
    def samples_per_window(self) -> int:
        return self.sfreq * self.window_seconds

    def to_json(self) -> dict:
        return asdict(self)


def feature_names(config: Config = Config()) -> list[str]:
    return [f"{ch}:{name}" for ch in config.channels for name in FEATURES_PER_CHANNEL]


def normalize_channel(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", name.upper().strip())


def select_channels(available: list[str], config: Config = Config()) -> list[str]:
    """Select exactly the prespecified montage, preserving its fixed order."""
    lookup: dict[str, str] = {}
    for ch in available:
        lookup.setdefault(normalize_channel(ch), ch)
    missing = [ch for ch in config.channels if normalize_channel(ch) not in lookup]
    if missing:
        raise ValueError(f"Missing mandatory EEG channels: {missing}")
    selected = [lookup[normalize_channel(ch)] for ch in config.channels]
    if len(set(selected)) != len(selected):
        raise ValueError("The channel mapping contains duplicates")
    return selected


def parse_summary(text: str) -> dict[str, dict]:
    """Parse official CHB-MIT chbXX-summary.txt text.

    Expects a seizure count in EVERY file block. No implicit zero seizures.
    """
    records: dict[str, dict] = {}
    current: dict | None = None
    starts: list[float] = []
    ends: list[float] = []

    def finish():
        if current is None:
            return
        count = current.get("declared_count")
        if count is None:
            raise ValueError(f"No seizure count for {current['filename']}")
        if len(starts) != count or len(ends) != count:
            raise ValueError(f"Incomplete seizure intervals for {current['filename']}")
        intervals = list(zip(starts, ends))
        if any(not (0 <= a < b) for a, b in intervals):
            raise ValueError(f"Invalid seizure interval for {current['filename']}")
        if any(b > c for (_, b), (c, _) in zip(intervals, intervals[1:])):
            raise ValueError(f"Unordered/overlapping seizures: {current['filename']}")
        key = current["filename"].lower()
        if key in records:
            raise ValueError(f"Duplicate annotation for {key}")
        records[key] = {"intervals": intervals, "declared_count": count}

    for original in text.splitlines():
        line = original.strip()
        match = re.match(r"File Name:\s*(\S+\.edf)\s*$", line, re.I)
        if match:
            finish()
            current = {"filename": match.group(1)}
            starts, ends = [], []
            continue
        if current is None:
            continue
        match = re.match(r"Number of Seizures in File:\s*(\d+)\s*$", line, re.I)
        if match:
            current["declared_count"] = int(match.group(1))
            continue
        match = re.match(r"Seizure(?:\s+\d+)?\s+(Start|End)\s+Time:\s*(\d+(?:\.\d+)?)\s+seconds?\s*$", line, re.I)
        if match:
            (starts if match.group(1).lower() == "start" else ends).append(float(match.group(2)))
    finish()
    return records


POSTICTAL_EXCLUSION_SECONDS = 30 * 60  # Research protocol choice; not a clinical standard.


def label_window(start: float, end: float, intervals: list[tuple[float, float]],
                 postictal_exclusion_seconds: float = POSTICTAL_EXCLUSION_SECONDS) -> int | None:
    """Return 1=fully ictal, 0=interictal candidate, None=excluded/ambiguous.

    Windows overlapping the configured postictal interval [seizure_end,
    seizure_end + exclusion) are excluded rather than mislabeled interictal.
    The 30-minute default is an explicit research-protocol choice and must not
    be presented as a clinically established recovery period.
    """
    if (not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start):
        raise ValueError("Invalid window timestamps")
    if (not math.isfinite(postictal_exclusion_seconds) or
            postictal_exclusion_seconds < 0):
        raise ValueError("Postictal exclusion must be finite and non-negative")

    previous_offset = -math.inf
    for onset, offset in intervals:
        if (not math.isfinite(onset) or not math.isfinite(offset) or
                onset < 0 or offset <= onset or onset < previous_offset):
            raise ValueError("Seizure intervals must be valid, sorted and non-overlapping")
        previous_offset = offset

    # Ictal labels take precedence only for windows fully contained in an event.
    for onset, offset in intervals:
        if start >= onset and end <= offset:
            return 1
        if start < offset and end > onset:
            return None

    # Any overlap with the postictal exclusion interval is unknown, not negative.
    if postictal_exclusion_seconds > 0 and any(
        start < offset + postictal_exclusion_seconds and end > offset
        for _, offset in intervals
    ):
        return None
    return 0


def validate_intervals(intervals: list[tuple[float, float]], duration: float) -> None:
    if any(a < 0 or b <= a or b > duration + 1 / 256 for a, b in intervals):
        raise ValueError("Seizure annotation falls outside the EDF recording")


def preprocess_window(signal: np.ndarray, config: Config = Config()) -> np.ndarray:
    """Filter EACH four-second window the same way in training and inference.

    Per-window filtering produces edge artifacts; do not interpret the first
    or last samples clinically. The choice guarantees parity across APIs.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.shape != (len(config.channels), config.samples_per_window):
        raise ValueError(f"Expected ({len(config.channels)}, {config.samples_per_window}) EEG samples")
    if not np.all(np.isfinite(x)):
        raise ValueError("EEG contains missing or non-finite samples")
    if not 0 < config.low_hz < config.high_hz < config.sfreq / 2:
        raise ValueError("Invalid bandpass configuration")
    if config.notch_hz is not None:
        raise ValueError("Notch filtering is not enabled: verify mains artifact before configuring it")
    sos = butter(4, [config.low_hz, config.high_hz], btype="bandpass", fs=config.sfreq, output="sos")
    filtered = sosfiltfilt(sos, x, axis=-1)
    if not np.all(np.isfinite(filtered)):
        raise ValueError("Filtering produced non-finite samples")
    return filtered


def extract_features(filtered_window: np.ndarray, config: Config = Config()) -> np.ndarray:
    x = np.asarray(filtered_window, dtype=np.float64)
    if x.shape != (len(config.channels), config.samples_per_window) or not np.all(np.isfinite(x)):
        raise ValueError("Expected finite, correctly shaped filtered EEG window")
    features: list[float] = []
    epsilon = 1e-30
    for ch in x:
        avg, var = float(np.mean(ch)), float(np.var(ch))
        d1, d2 = np.diff(ch), np.diff(ch, n=2)
        v1 = float(np.var(d1))
        mobility = math.sqrt(v1 / max(var, epsilon))
        d2_mobility = math.sqrt(float(np.var(d2)) / max(v1, epsilon))
        complexity = d2_mobility / max(mobility, 1e-15)
        freqs, power = welch(ch, fs=config.sfreq, nperseg=min(512, len(ch)), scaling="density")
        bandpowers = []
        for low, high in BANDS.values():
            mask = (freqs >= low) & (freqs <= high)
            bandpowers.append(float(np.trapezoid(power[mask], freqs[mask])) if mask.sum() >= 2 else 0.0)
        total_power = max(sum(bandpowers), epsilon)
        rel = [v / total_power for v in bandpowers]
        spectrum = power[(freqs >= config.low_hz) & (freqs <= config.high_hz)]
        normalized = spectrum / max(float(np.sum(spectrum)), epsilon)
        entropy = -float(np.sum(normalized * np.log2(np.maximum(normalized, 1e-30))))
        entropy /= math.log2(len(normalized)) if len(normalized) > 1 else 1.0
        features.extend([
            avg, math.sqrt(var), var, float(np.sqrt(np.mean(ch * ch))),
            float(np.sum(ch * ch)), float(np.ptp(ch)),
            float(np.mean(np.signbit(ch[1:]) != np.signbit(ch[:-1]))),
            mobility, complexity, *bandpowers, *rel, entropy,
        ])
    result = np.asarray(features, dtype=np.float64)
    if len(result) != len(feature_names(config)) or not np.all(np.isfinite(result)):
        raise ValueError("Feature extraction returned invalid values")
    return result


def load_edf(path: str | Path):
    try:
        import mne
    except ImportError as exc:
        raise RuntimeError("MNE is required for reading EDF: pip install -r requirements.txt") from exc
    return mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR")


def discover_recordings(data_dir: str | Path, config: Config = Config()) -> dict:
    """Audit the actual data. Missing annotations never imply 'interictal'."""
    root = Path(data_dir)
    files = sorted(root.rglob("*.edf")) if root.is_dir() else []
    audit: dict = {
        "status": "blocked_missing_dataset" if not root.is_dir() else "audited",
        "dataset_path": str(root), "edf_files_found": len(files), "edf_files_readable": 0,
        "patients": [], "verified_seizures": 0, "recording_hours": 0.0,
        "ictal_windows": 0, "interictal_windows": 0, "boundary_windows_excluded": 0,
        "missing_or_invalid_annotations": [], "rejected_recordings": [],
        "accepted_recordings": [], "class_distribution": {},
        "annotation_source": "CHB-MIT patient summary text, checked against EDF duration",
        "channels": list(config.channels), "sfreq_hz": config.sfreq,
        "window_seconds": config.window_seconds,
    }
    if not files:
        if root.is_dir():
            audit["status"] = "blocked_no_edf"
        return audit

    summary_cache = {}
    for path in files:
        patient_match = re.match(r"^(chb\d+)_", path.name.lower())
        patient = patient_match.group(1) if patient_match else None
        if not patient:
            audit["rejected_recordings"].append({"file": str(path), "reason": "Unrecognized patient ID"})
            continue
        summary = next((p for p in [path.parent / f"{patient}-summary.txt", root / f"{patient}-summary.txt"] if p.is_file()), None)
        if summary is None:
            audit["missing_or_invalid_annotations"].append(str(path))
            audit["rejected_recordings"].append({"file": str(path), "reason": "Patient summary missing"})
            continue
        try:
            if summary not in summary_cache:
                summary_cache[summary] = parse_summary(summary.read_text(encoding="utf-8-sig"))
            entry = summary_cache[summary].get(path.name.lower())
            if entry is None:
                raise ValueError("File is absent from patient summary")
            raw = load_edf(path)
            try:
                audit["edf_files_readable"] += 1
                if not np.isclose(raw.info["sfreq"], config.sfreq):
                    raise ValueError(f"Sampling frequency {raw.info['sfreq']} Hz differs from {config.sfreq} Hz")
                ch = select_channels(raw.ch_names, config)
                duration = raw.n_times / raw.info["sfreq"]
                validate_intervals(entry["intervals"], duration)
            finally:
                raw.close()
            counts = {"ictal": 0, "interictal": 0, "boundary": 0}
            for i in range(math.floor(duration / config.window_seconds)):
                start = i * config.window_seconds
                label = label_window(start, start + config.window_seconds, entry["intervals"])
                counts["boundary" if label is None else ("ictal" if label else "interictal")] += 1
            audit["verified_seizures"] += len(entry["intervals"])
            audit["recording_hours"] += duration / 3600.0
            audit["ictal_windows"] += counts["ictal"]
            audit["interictal_windows"] += counts["interictal"]
            audit["boundary_windows_excluded"] += counts["boundary"]
            audit["accepted_recordings"].append({
                "file": str(path), "patient": patient, "duration_seconds": duration,
                "channels_actual": ch, "seizures": entry["intervals"], "window_counts": counts,
            })
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            audit["rejected_recordings"].append({"file": str(path), "reason": reason})
            if "summary" in reason.lower() or "seizure" in reason.lower() or "annotation" in reason.lower():
                audit["missing_or_invalid_annotations"].append(str(path))

    audit["patients"] = sorted({r["patient"] for r in audit["accepted_recordings"]})
    audit["class_distribution"] = {"interictal": audit["interictal_windows"], "ictal": audit["ictal_windows"]}
    if not audit["accepted_recordings"]:
        audit["status"] = "blocked_no_valid_recordings"
    return audit


def iter_record_windows(record: dict, config: Config = Config(),
                        rejected_windows: list[dict] | None = None):
    raw = load_edf(record["file"])
    try:
        channels = select_channels(raw.ch_names, config)
        n_windows = math.floor((raw.n_times / raw.info["sfreq"]) / config.window_seconds)
        for idx in range(n_windows):
            start = idx * config.window_seconds
            end = start + config.window_seconds
            target = label_window(start, end, record["seizures"])
            if target is None:
                if rejected_windows is not None:
                    rejected_windows.append({"file": record["file"], "start_seconds": start,
                                             "end_seconds": end, "reason": "crosses_seizure_boundary"})
                continue
            first = idx * config.samples_per_window
            try:
                x = raw.get_data(picks=channels, start=first, stop=first + config.samples_per_window)
                features = extract_features(preprocess_window(x, config), config)
            except (ValueError, RuntimeError, OSError) as exc:
                if rejected_windows is not None:
                    rejected_windows.append({"file": record["file"], "start_seconds": start,
                                             "end_seconds": end, "reason": f"invalid_eeg: {exc}"})
                continue
            yield features, target, {"file": record["file"], "patient": record["patient"],
                                     "start_seconds": float(start), "end_seconds": float(end)}
    finally:
        raw.close()


def compare_filtering_methods(signal: np.ndarray, config: Config = Config()) -> dict:
    """Quantify the boundary artifact from isolated-window filtering.

    Input must be a finite CONTINUOUS signal containing >= 3 full windows.
    This is a diagnostic; it does not silently change the training pipeline.
    """
    samples = np.asarray(signal, dtype=np.float64)
    length = config.samples_per_window
    if samples.ndim != 2 or samples.shape[0] != len(config.channels) or samples.shape[1] < 3 * length:
        raise ValueError("Filtering comparison needs >= 3 contiguous four-second windows")
    if not np.isfinite(samples).all():
        raise ValueError("Signal contains non-finite values")
    if config.notch_hz is not None:
        raise ValueError("Notch requires independent mains-artifact validation")
    sos = butter(4, [config.low_hz, config.high_hz], btype="bandpass",
                 fs=config.sfreq, output="sos")
    continuous = sosfiltfilt(sos, samples, axis=-1)
    # Compare an interior window, where continuous filtering has signal context.
    middle = samples[:, length:2 * length]
    separately = preprocess_window(middle, config)
    reference = continuous[:, length:2 * length]
    delta = separately - reference
    return {"rms_difference_volts": float(np.sqrt(np.mean(delta ** 2))),
            "max_absolute_difference_volts": float(np.max(np.abs(delta))),
            "note": "Diagnostic only; training/inference currently filter individual windows"}


def split_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Explicit patient-independent baseline: CHB01, CHB02 -> CHB03."""
    train = [r for r in records if r["patient"] in ("chb01", "chb02")]
    test = [r for r in records if r["patient"] == "chb03"]
    if not train or not test:
        raise ValueError("Need labeled CHB01/CHB02 training and CHB03 test recordings")
    if set(r["patient"] for r in train) & set(r["patient"] for r in test):
        raise AssertionError("Patient leakage")
    return train, test


# PhysioNet documents CHB-MIT cases chb01 and chb21 as recordings from
# the same participant, acquired approximately 1.5 years apart. Keep them
# together for participant-independent evaluation.
SUBJECT_ID_BY_CASE = {
    "chb01": "subject_chb01_chb21",
    "chb21": "subject_chb01_chb21",
}


def subject_id_for_case(patient_id: str) -> str:
    case_id = str(patient_id).strip().lower()
    return SUBJECT_ID_BY_CASE.get(case_id, case_id)


def split_records_three_way(records: list[dict],
                            split_patients: dict[str, list[str]] | None = None
                            ) -> tuple[list[dict], list[dict], list[dict]]:
    """Disjoint participant-level train/validation/test; no window shuffling.

    Explicit split lists contain CHB-MIT case IDs (chbXX), but all cases
    belonging to one real participant must remain in the same partition.
    """
    available = sorted({str(r["patient"]).strip().lower() for r in records})
    if not available:
        raise ValueError("No patient records are available")
    subject_to_cases: dict[str, list[str]] = {}
    for case_id in available:
        subject_to_cases.setdefault(subject_id_for_case(case_id), []).append(case_id)
    subject_groups = sorted(subject_to_cases)
    if len(subject_groups) < 3:
        raise ValueError("Need at least 3 distinct participants for train/validation/test")

    if split_patients is None:
        train_subjects = subject_groups[:-2]
        validation_subjects = [subject_groups[-2]]
        test_subjects = [subject_groups[-1]]
        chosen = {
            "train": [case for subject in train_subjects for case in subject_to_cases[subject]],
            "validation": subject_to_cases[validation_subjects[0]],
            "test": subject_to_cases[test_subjects[0]],
        }
    else:
        if not isinstance(split_patients, dict) or set(split_patients) != {"train", "validation", "test"}:
            raise ValueError("Split must define exactly train, validation and test groups")
        chosen = {}
        for name in ("train", "validation", "test"):
            cases = split_patients[name]
            if not isinstance(cases, list) or not cases or any(
                    not isinstance(case, str) or not case.strip() for case in cases):
                raise ValueError(f"Split group {name!r} must be a non-empty list of case IDs")
            normalized = [case.strip().lower() for case in cases]
            if len(normalized) != len(set(normalized)):
                raise ValueError(f"Split group {name!r} contains duplicate case IDs")
            chosen[name] = normalized

    sets = [set(chosen[name]) for name in ("train", "validation", "test")]
    if any(not group for group in sets):
        raise ValueError("Every split must contain at least one case")
    if any(a & b for i, a in enumerate(sets) for b in sets[i + 1:]):
        raise ValueError("Case leakage between dataset splits")
    if set.union(*sets) != set(available):
        raise ValueError("Split does not cover exactly the audited case IDs")

    partition_subjects = [
        {subject_id_for_case(case_id) for case_id in group}
        for group in sets
    ]
    if any(a & b for i, a in enumerate(partition_subjects)
           for b in partition_subjects[i + 1:]):
        raise ValueError("Participant leakage: related CHB-MIT cases cross dataset splits")

    return tuple([r for r in records
                  if str(r["patient"]).strip().lower() in group] for group in sets)
