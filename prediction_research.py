"""FUTURE RESEARCH ONLY: candidate preictal labeling, not a trained predictor.

Experiment proposal: assess whether EEG windows between 30 and 10 minutes
before a future seizure carry a discriminative signal. These thresholds are
research choices, not clinically verified warning times.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionProtocol:
    outer_horizon_seconds: int = 30 * 60
    minimum_warning_seconds: int = 10 * 60
    postictal_exclusion_seconds: int = 30 * 60

    def __post_init__(self):
        if not 0 < self.minimum_warning_seconds < self.outer_horizon_seconds:
            raise ValueError("Prediction horizon must be larger than the minimum warning time")


def candidate_preictal_label(start: float, end: float,
                             seizures: list[tuple[float, float]],
                             protocol: PredictionProtocol = PredictionProtocol(),
                             *, continuous_intervals: list[tuple[float, float]] | None = None,
                             annotation_verified_through: float | None = None) -> int | None:
    """0=eligible interictal; 1=candidate preictal; None=excluded/ambiguous.

    A window qualifies only when entirely inside the research preictal range
    for an event in this SAME continuous recording. Recording gaps, unverified
    event annotations, and seizure-free future data cannot be assumed away.
    """
    if start < 0 or end <= start:
        raise ValueError("Invalid window timestamps")
    # A genuine negative requires observing the entire future prediction
    # horizon. Never equate 'no known seizure' with a confirmed negative.
    future_end = end + protocol.outer_horizon_seconds
    if continuous_intervals is None or annotation_verified_through is None:
        return None
    if future_end > annotation_verified_through:
        return None
    if not any(a <= start and future_end <= b for a, b in continuous_intervals):
        return None  # Unrecorded future, a recording boundary, or a gap.
    for onset, offset in seizures:
        if start < offset + protocol.postictal_exclusion_seconds and end > onset:
            return None  # ictal and postictal/near-ictal never become negatives
    candidate = False
    for onset, _ in seizures:
        left = onset - protocol.outer_horizon_seconds
        right = onset - protocol.minimum_warning_seconds
        if start >= left and end <= right:
            candidate = True
        elif start < onset and end > left:
            return None  # Crossing a preictal or embargo boundary is not safe
    return 1 if candidate else 0


RESEARCH_WARNINGS = (
    "These labels are provisional research hypotheses and are not validated warning times.",
    "Patient-independent evaluation and event-level false alarms/day are required.",
    "A clinically useful prediction horizon cannot be asserted without prospective validation.",
    "Do not train on disconnected recording gaps or unlabeled periods as if verified negative.",
)
