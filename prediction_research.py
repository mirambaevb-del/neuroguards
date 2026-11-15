"""FUTURE RESEARCH ONLY: candidate preictal labeling, not a trained predictor.

Experiment proposal: assess whether EEG windows between 30 and 10 minutes
before a future seizure carry a discriminative signal. These thresholds are
research choices, not clinically verified warning times.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PredictionProtocol:
    outer_horizon_seconds: int = 30 * 60
    minimum_warning_seconds: int = 10 * 60
    postictal_exclusion_seconds: int = 30 * 60

    def __post_init__(self):
        if self.outer_horizon_seconds <= 0:
            raise ValueError("Outer prediction horizon must be positive")
        if not 0 < self.minimum_warning_seconds < self.outer_horizon_seconds:
            raise ValueError("Minimum warning time must be positive and smaller than the outer horizon")
        if self.postictal_exclusion_seconds < 0:
            raise ValueError("Postictal exclusion must not be negative")


def _covered_by_one_interval(start: float, end: float,
                             intervals: list[tuple[float, float]]) -> bool:
    """Require one continuous recording segment to cover the complete span."""
    return any(left <= start and end <= right for left, right in intervals)


def candidate_preictal_label(start: float, end: float,
                             seizures: list[tuple[float, float]],
                             protocol: PredictionProtocol = PredictionProtocol(),
                             *, continuous_intervals: list[tuple[float, float]] | None = None,
                             annotation_verified_through: float | None = None) -> int | None:
    """Return 0=verified negative, 1=candidate preictal, None=excluded/unknown.

    A positive label only needs continuous EEG and verified seizure annotation
    through the qualifying seizure onset. It must NOT require another full
    30-minute observation period after that onset.

    A negative label is stricter: the recording and annotation must both
    continue through the complete future outer horizon, with no seizure in it.
    Gaps are never bridged by combining separate continuous intervals.
    """
    values = (start, end)
    if not all(math.isfinite(value) for value in values) or start < 0 or end <= start:
        raise ValueError("Invalid window timestamps")
    if continuous_intervals is None or annotation_verified_through is None:
        return None
    if not math.isfinite(annotation_verified_through) or annotation_verified_through < 0:
        raise ValueError("annotation_verified_through must be finite and non-negative")

    for left, right in continuous_intervals:
        if not math.isfinite(left) or not math.isfinite(right) or left < 0 or right <= left:
            raise ValueError("Invalid continuous recording interval")
    for onset, offset in seizures:
        if (not math.isfinite(onset) or not math.isfinite(offset) or
                onset < 0 or offset <= onset):
            raise ValueError("Invalid seizure interval")

    # The EEG window itself must be inside one uninterrupted recording segment.
    if not _covered_by_one_interval(start, end, continuous_intervals):
        return None

    candidate_events: list[float] = []
    for onset, offset in seizures:
        # Never label ictal, postictal, or near-ictal windows as interictal.
        if start < offset + protocol.postictal_exclusion_seconds and end > onset:
            return None

        preictal_left = onset - protocol.outer_horizon_seconds
        preictal_right = onset - protocol.minimum_warning_seconds
        overlaps_preictal = start < preictal_right and end > preictal_left
        fully_inside_preictal = start >= preictal_left and end <= preictal_right

        if overlaps_preictal and not fully_inside_preictal:
            return None  # Window crosses a research-label boundary.
        if fully_inside_preictal:
            candidate_events.append(onset)

    # A candidate positive is confirmed by observing the uninterrupted EEG
    # through its onset and having verified annotations through that onset.
    for onset in candidate_events:
        if (annotation_verified_through >= onset and
                _covered_by_one_interval(start, onset, continuous_intervals)):
            return 1

    # If a seizure occurs within the future outer horizon but this window did
    # not qualify as a complete preictal window, its class is ambiguous, not 0.
    future_end = end + protocol.outer_horizon_seconds
    if any(end < onset <= future_end for onset, _ in seizures):
        return None

    # A true negative needs full continuous EEG + annotation coverage through
    # the entire horizon. This check is deliberately applied only to negatives.
    if annotation_verified_through < future_end:
        return None
    if not _covered_by_one_interval(start, future_end, continuous_intervals):
        return None
    return 0


RESEARCH_WARNINGS = (
    "These labels are provisional research hypotheses and are not validated warning times.",
    "Patient-independent evaluation and event-level false alarms/day are required.",
    "A clinically useful prediction horizon cannot be asserted without prospective validation.",
    "Do not train on disconnected recording gaps or unlabeled periods as if verified negative.",
)
