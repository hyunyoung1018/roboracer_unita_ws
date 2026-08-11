"""Lightweight, position-history obstacle classification.

This module deliberately has no ROS dependency so the classification rules can
be unit-tested without starting an executor.  It borrows only the circular
position statistics idea from UNICORN's tracker; association and motion
estimation stay in UNITA's existing tracking node.
"""

from collections import deque
from dataclasses import dataclass, field
import math


UNKNOWN = "UNKNOWN"
STATIC = "STATIC"
DYNAMIC = "DYNAMIC"


def circular_delta(a, b, track_length):
    """Return the shortest signed displacement from ``b`` to ``a``."""
    return (a - b + track_length / 2.0) % track_length - track_length / 2.0


def population_std(samples):
    """Small population-standard-deviation helper with no NumPy dependency."""
    if not samples:
        return 0.0
    mean = sum(samples) / len(samples)
    variance = sum((value - mean) ** 2 for value in samples) / len(samples)
    return math.sqrt(variance)


def circular_std(samples, track_length):
    """Compute seam-safe s variation by unwrapping about the latest sample."""
    if not samples:
        return 0.0
    reference = samples[-1]
    unwrapped = [
        reference + circular_delta(value, reference, track_length)
        for value in samples
    ]
    return population_std(unwrapped)


def output_membership(stable_class):
    """Return ``(in_static_output, in_dynamic_output)`` for a stable class."""
    return stable_class == STATIC, stable_class == DYNAMIC


@dataclass
class TrackHistory:
    """Bounded classification state belonging to exactly one tracker ID."""

    obstacle_id: int
    s_history: deque
    d_history: deque
    timestamp_history: deque
    stable_class: str = UNKNOWN
    static_evidence: int = 0
    dynamic_evidence: int = 0
    last_seen: float = 0.0
    std_s: float = 0.0
    std_d: float = 0.0

    @property
    def sample_count(self):
        return len(self.s_history)


@dataclass
class StableObstacleClassifier:
    """Classify histories using circular std and small hysteresis counters."""

    min_std: float = 0.02
    max_std: float = 0.04
    min_nb_meas: int = 8
    history_size: int = 20
    static_confirm_count: int = 3
    dynamic_confirm_count: int = 2
    track_timeout_sec: float = 1.0
    track_length: float = None
    tracks: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.min_std < 0.0 or self.max_std <= self.min_std:
            raise ValueError("Require 0 <= min_std < max_std")
        if self.min_nb_meas < 2 or self.history_size < self.min_nb_meas:
            raise ValueError("Require history_size >= min_nb_meas >= 2")
        if self.static_confirm_count < 1 or self.dynamic_confirm_count < 1:
            raise ValueError("Confirmation counts must be positive")
        if self.track_timeout_sec <= 0.0:
            raise ValueError("track_timeout_sec must be positive")

    def _new_track(self, obstacle_id, now):
        history = TrackHistory(
            obstacle_id=int(obstacle_id),
            s_history=deque(maxlen=self.history_size),
            d_history=deque(maxlen=self.history_size),
            timestamp_history=deque(maxlen=self.history_size),
            last_seen=float(now),
        )
        self.tracks[int(obstacle_id)] = history
        return history

    def update(self, obstacle_id, s, d, is_visible, now):
        """Update one ID and return its classification state.

        Retained, invisible tracker frames refresh ``last_seen`` but are not
        treated as new measurements and cannot add classification evidence.
        """
        obstacle_id = int(obstacle_id)
        history = self.tracks.get(obstacle_id)
        if history is None:
            history = self._new_track(obstacle_id, now)
        history.last_seen = float(now)

        if not is_visible:
            return history

        history.s_history.append(float(s))
        history.d_history.append(float(d))
        history.timestamp_history.append(float(now))

        if history.sample_count < self.min_nb_meas:
            return history
        if self.track_length is None or self.track_length <= 0.0:
            return history

        history.std_s = circular_std(history.s_history, self.track_length)
        history.std_d = population_std(history.d_history)

        clearly_static = (
            history.std_s < self.min_std
            and history.std_d < self.min_std
        )
        clearly_dynamic = (
            history.std_s > self.max_std
            or history.std_d > self.max_std
        )

        if clearly_static:
            history.static_evidence = min(
                history.static_evidence + 1, self.static_confirm_count)
            history.dynamic_evidence = 0
            if history.static_evidence >= self.static_confirm_count:
                history.stable_class = STATIC
        elif clearly_dynamic:
            history.dynamic_evidence = min(
                history.dynamic_evidence + 1, self.dynamic_confirm_count)
            history.static_evidence = 0
            if history.dynamic_evidence >= self.dynamic_confirm_count:
                history.stable_class = DYNAMIC
        else:
            # Ambiguous evidence never changes the stable class.  Resetting the
            # counters makes confirmation mean consecutive clear observations.
            history.static_evidence = 0
            history.dynamic_evidence = 0

        return history

    def remove_stale(self, now):
        """Drop IDs absent from the raw stream beyond the configured TTL."""
        stale_ids = [
            obstacle_id
            for obstacle_id, history in self.tracks.items()
            if float(now) - history.last_seen > self.track_timeout_sec
        ]
        for obstacle_id in stale_ids:
            del self.tracks[obstacle_id]
        return stale_ids
