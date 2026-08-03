"""Deterministic, identity-safe association sampling primitives."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

from .types import Detection3D, SamplingType


@dataclass(frozen=True)
class SampleSelection:
    source_id: str
    detection: Detection3D
    sampling_type: SamplingType
    distance_m: float
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class JitterResult:
    detection: Detection3D
    sampling_type: SamplingType
    original_geometry_7d: Tuple[float, ...]
    jittered_geometry_7d: Tuple[float, ...]
    noise_7d: Tuple[float, ...]
    standard_deviation: float
    seed: int


def _distance(first: Detection3D, second: Detection3D) -> float:
    return math.dist(first.position_m, second.position_m)


def select_negative_candidate(
    history_track_id: str,
    anchor: Detection3D,
    candidates: Sequence[Tuple[str, Detection3D]],
    *,
    strategy: str,
    seed: int,
    hard_k: int = 3,
    local_radius_m: Optional[float] = None,
) -> SampleSelection:
    """Select a different-identity negative using an explicit strategy."""

    if not isinstance(history_track_id, str) or not history_track_id.strip():
        raise ValueError("history_track_id must be non-empty")
    eligible = tuple((source_id, detection) for source_id, detection in candidates if source_id != history_track_id)
    if not eligible:
        raise ValueError("negative candidate set is empty after identity exclusion")
    distances = tuple(_distance(anchor, detection) for _, detection in eligible)
    generator = random.Random(seed)
    if strategy == "hard":
        if isinstance(hard_k, bool) or not isinstance(hard_k, int) or hard_k <= 0:
            raise ValueError("hard_k must be a positive integer")
        ranked = sorted(range(len(eligible)), key=lambda index: (distances[index], eligible[index][0]))
        pool = ranked[: min(hard_k, len(ranked))]
        selected = generator.choice(pool)
        sampling_type = SamplingType.NEGATIVE_HARD
    elif strategy == "local":
        if local_radius_m is None or not math.isfinite(local_radius_m) or local_radius_m < 0:
            raise ValueError("local_radius_m must be explicit and non-negative")
        pool = [index for index, distance in enumerate(distances) if distance <= local_radius_m]
        if not pool:
            raise ValueError("local negative set is empty; no implicit fallback is allowed")
        selected = generator.choice(pool)
        sampling_type = SamplingType.NEGATIVE_LOCAL
    elif strategy == "random":
        selected = generator.randrange(len(eligible))
        sampling_type = SamplingType.NEGATIVE_RANDOM
    else:
        raise ValueError("strategy must be hard, local, or random")
    source_id, detection = eligible[selected]
    return SampleSelection(
        source_id=source_id,
        detection=detection,
        sampling_type=sampling_type,
        distance_m=distances[selected],
        metadata={
            "strategy": strategy,
            "seed": seed,
            "hard_k": hard_k if strategy == "hard" else None,
            "local_radius_m": local_radius_m if strategy == "local" else None,
            "eligible_count": len(eligible),
        },
    )


def jitter_positive_detection(
    detection: Detection3D,
    *,
    standard_deviation: float,
    seed: int,
) -> JitterResult:
    """Apply the evidence-backed Gaussian profile to normalized xyz/lwh/yaw."""

    if (
        isinstance(standard_deviation, bool)
        or not isinstance(standard_deviation, (int, float))
        or not math.isfinite(float(standard_deviation))
        or standard_deviation < 0
    ):
        raise ValueError("standard_deviation must be finite and non-negative")
    generator = random.Random(seed)
    noise = tuple(generator.gauss(0.0, float(standard_deviation)) for _ in range(7))
    normalized = (
        detection.position_m[0] / 10.0,
        detection.position_m[1] / 10.0,
        detection.position_m[2] / 10.0,
        detection.size_lwh_m[0],
        detection.size_lwh_m[1],
        detection.size_lwh_m[2],
        detection.yaw_rad,
    )
    jittered = tuple(value + delta for value, delta in zip(normalized, noise))
    if any(value <= 0 for value in jittered[3:6]):
        raise ValueError("jitter produced a non-positive box dimension")
    result_detection = Detection3D(
        detection_id=detection.detection_id,
        frame=detection.frame,
        position_m=(jittered[0] * 10.0, jittered[1] * 10.0, jittered[2] * 10.0),
        size_lwh_m=(jittered[3], jittered[4], jittered[5]),
        yaw_rad=jittered[6],
        score=detection.score,
        category=detection.category,
        coordinate_frame=detection.coordinate_frame,
        class_membership=detection.class_membership,
    )
    return JitterResult(
        detection=result_detection,
        sampling_type=SamplingType.POSITIVE_JITTERED,
        original_geometry_7d=normalized,
        jittered_geometry_7d=jittered,
        noise_7d=noise,
        standard_deviation=float(standard_deviation),
        seed=seed,
    )
