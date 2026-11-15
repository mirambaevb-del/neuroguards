"""Reusable NeuroGuard EEG inference (no training). Research use only."""
from __future__ import annotations

import json
from pathlib import Path
import hashlib

import joblib
import numpy as np

from neuroguard_core import (Config, VERSION, extract_features, feature_names,
                             load_edf, preprocess_window, select_channels)


class NeuroGuardDetector:
    def __init__(self, model_dir: str | Path | None = None):
        base = Path(model_dir) if model_dir is not None else Path(__file__).resolve().parent / "model"
        required = [base / item for item in ("seizure_detector.joblib", "config.json", "features.json", "manifest.json")]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"No trained model is available. Missing: {missing}. Train on real annotated EDF files first.")
        # Only load locally trained, trusted joblib artifacts; joblib uses pickle.
        conf = json.loads((base / "config.json").read_text(encoding="utf-8"))
        self.config = Config(**{**conf, "channels": tuple(conf["channels"])})
        self.features_metadata = json.loads((base / "features.json").read_text(encoding="utf-8"))
        self.manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
        if self.features_metadata["feature_names"] != feature_names(self.config):
            raise RuntimeError("Stored model feature order differs from preprocessing configuration")
        expected_hash = self.manifest.get("model_sha256")
        if expected_hash is not None:
            actual_hash = hashlib.sha256((base / "seizure_detector.joblib").read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                raise RuntimeError("Model checksum mismatch: artifact may be damaged or incompatible")
        self.model = joblib.load(base / "seizure_detector.joblib")
        if getattr(self.model, "n_features_in_", None) != len(feature_names(self.config)):
            raise RuntimeError("Trained model feature count does not match extraction code")

    def _predict_window(self, window: np.ndarray, start: float) -> dict:
        feature = extract_features(preprocess_window(window, self.config), self.config).reshape(1, -1)
        label = int(self.model.predict(feature)[0])
        score = None
        if hasattr(self.model, "predict_proba"):
            classes = list(self.model.classes_)
            if 1 in classes:
                score = float(self.model.predict_proba(feature)[0, classes.index(1)])
        return {
            "label": "ICTAL" if label == 1 else "INTERICTAL", "class_id": label,
            "seizure_score_uncalibrated": score, "calibrated_probability": None,
            "start_seconds": float(start), "end_seconds": float(start + self.config.window_seconds),
            "model_version": self.manifest.get("version", VERSION), "status": "ok",
        }

    def predict_array(self, samples: np.ndarray, channels: list[str], sfreq: float,
                      start_seconds: float = 0.0, *, unit: str) -> dict:
        """Samples: channels x time, in VOLTS, in requested channel names."""
        if unit != "V":
            raise ValueError("EEG unit must be explicitly specified as 'V' (volts)")
        if not np.isclose(sfreq, self.config.sfreq):
            raise ValueError(f"Sampling frequency must be {self.config.sfreq} Hz")
        arr = np.asarray(samples, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != len(channels):
            raise ValueError("EEG shape must be [number_of_channels, number_of_samples]")
        selected = select_channels(channels, self.config)
        idx = [channels.index(ch) for ch in selected]
        arr = arr[idx]
        if not np.all(np.isfinite(arr)):
            raise ValueError("EEG has NaN/Infinity/missing samples")
        n = arr.shape[1] // self.config.samples_per_window
        if n <= 0 or n * self.config.samples_per_window != arr.shape[1]:
            raise ValueError(f"EEG length must be a positive multiple of {self.config.samples_per_window} samples")
        if not np.isfinite(start_seconds) or start_seconds < 0:
            raise ValueError("start_seconds must be nonnegative and finite")
        predictions = [self._predict_window(arr[:, i * self.config.samples_per_window:
                                                 (i + 1) * self.config.samples_per_window],
                                            start_seconds + i * self.config.window_seconds)
                       for i in range(n)]
        return {"status": "ok", "model_version": self.manifest.get("version", VERSION),
                "task": "SEIZURE_DETECTION_NOT_PREDICTION", "windows": predictions}

    def predict_edf(self, path: str | Path, max_windows: int = 3600) -> dict:
        """Process an EDF file in streaming 4-second windows."""
        raw = load_edf(path)
        try:
            if not np.isclose(raw.info["sfreq"], self.config.sfreq):
                raise ValueError(f"EDF has {raw.info['sfreq']} Hz; expected {self.config.sfreq} Hz")
            channels = select_channels(raw.ch_names, self.config)
            n = raw.n_times // self.config.samples_per_window
            if n > max_windows:
                raise ValueError(f"Recording has {n} windows; limit is {max_windows}")
            windows = []
            for i in range(n):
                first = i * self.config.samples_per_window
                chunk = raw.get_data(picks=channels, start=first, stop=first + self.config.samples_per_window)
                try:
                    windows.append(self._predict_window(chunk, i * self.config.window_seconds))
                except ValueError as exc:
                    windows.append({"start_seconds": i * self.config.window_seconds,
                                    "end_seconds": (i + 1) * self.config.window_seconds,
                                    "status": "invalid_window", "error": str(exc)})
            return {"status": "ok", "model_version": self.manifest.get("version", VERSION),
                    "task": "SEIZURE_DETECTION_NOT_PREDICTION", "windows": windows,
                    "remainder_samples_not_scored": int(raw.n_times % self.config.samples_per_window)}
        finally:
            raw.close()


class StreamingEEGSession:
    """Retain incomplete windows across transport packets in ONE session.

    The session is intentionally not shared across patients or requests.
    Caller is responsible for chronological packets without sample gaps.
    """
    def __init__(self, detector: NeuroGuardDetector, start_seconds: float = 0.0):
        self.detector = detector
        self.start_seconds = float(start_seconds)
        if not np.isfinite(self.start_seconds) or self.start_seconds < 0:
            raise ValueError("start_seconds must be nonnegative and finite")
        self.next_packet_seconds = self.start_seconds
        self.buffer = np.empty((len(detector.config.channels), 0), dtype=np.float64)
        self.closed = False

    def push(self, samples: np.ndarray, channels: list[str], sfreq: float, *,
             unit: str, packet_start_seconds: float) -> list[dict]:
        """Append one packet whose timestamp must follow the previous packet exactly.

        packet_start_seconds is the time of the first sample in this packet,
        relative to the beginning of the EDF/session. Packets with gaps,
        overlaps, or out-of-order timestamps are rejected.
        """
        if self.closed:
            raise ValueError("Stream already finalized")
        if unit != "V" or not np.isclose(sfreq, self.detector.config.sfreq):
            raise ValueError("Streaming EEG requires explicit volts and matching sampling rate")
        if not np.isfinite(packet_start_seconds) or packet_start_seconds < 0:
            raise ValueError("packet_start_seconds must be nonnegative and finite")
        if not np.isclose(packet_start_seconds, self.next_packet_seconds,
                          rtol=0.0, atol=0.5 / self.detector.config.sfreq):
            raise ValueError(
                f"Non-contiguous EEG packet: expected start at {self.next_packet_seconds:.9f}s, "
                f"received {packet_start_seconds:.9f}s"
            )
        x = np.asarray(samples, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] != len(channels):
            raise ValueError("Streaming samples must have shape channels x time")
        actual = select_channels(channels, self.detector.config)
        ordered = x[[channels.index(name) for name in actual], :]
        if ordered.shape[1] == 0:
            raise ValueError("Streaming EEG packet must contain at least one sample")
        if not np.isfinite(ordered).all():
            raise ValueError("Streaming EEG contains NaN or Infinity")
        self.buffer = np.concatenate((self.buffer, ordered), axis=1)
        # Advance from the expected timeline, not the received timestamp, so tolerated
        # sub-sample timestamp jitter cannot accumulate into long-term clock drift.
        self.next_packet_seconds += ordered.shape[1] / self.detector.config.sfreq
        n = self.buffer.shape[1] // self.detector.config.samples_per_window
        outputs = []
        length = self.detector.config.samples_per_window
        for i in range(n):
            outputs.append(self.detector._predict_window(self.buffer[:, i * length:(i + 1) * length],
                                                        self.start_seconds + i * self.detector.config.window_seconds))
        self.buffer = self.buffer[:, n * length:]
        self.start_seconds += n * self.detector.config.window_seconds
        return outputs

    def finalize(self) -> dict:
        self.closed = True
        return {"status": "closed", "unscored_remainder_samples": int(self.buffer.shape[1]),
                "next_window_start_seconds": self.start_seconds}


def temporal_confirmation(windows: list[dict], consecutive_positives: int = 2) -> list[dict]:
    """Research-only debounce; may increase detection delay and miss short events."""
    if consecutive_positives < 1:
        raise ValueError("consecutive_positives must be positive")
    run = 0
    result = []
    for w in windows:
        run = run + 1 if w.get("label") == "ICTAL" else 0
        result.append({**w, "temporally_confirmed_ictal": run >= consecutive_positives})
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("edf")
    p.add_argument("--model-dir", default=None)
    args = p.parse_args()
    print(json.dumps(NeuroGuardDetector(args.model_dir).predict_edf(args.edf), indent=2))
