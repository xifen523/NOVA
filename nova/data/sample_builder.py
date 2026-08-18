"""Typed training-sample builder integrating prompting, mining, and jitter."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Mapping, Optional, Sequence

from nova.preprocessing.association import ClassPromptPolicy, serialize_association_context

from .base import DatasetAdapter
from .sampling import jitter_positive_detection, select_negative_candidate
from .types import (
    AssociationCandidate,
    AssociationMode,
    AssociationSample,
    Detection3D,
    SamplingType,
    TrajectoryHistory,
)


def allocate_sampling_counts(total: int, ratios: Mapping[str, float]) -> Dict[str, int]:
    """Allocate an explicit mixture deterministically using largest remainders."""

    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        raise ValueError("total must be a positive integer")
    if not ratios or abs(sum(float(value) for value in ratios.values()) - 1.0) > 1e-8:
        raise ValueError("sampling ratios must be explicit and sum to 1")
    if any(value < 0 for value in ratios.values()):
        raise ValueError("sampling ratios must be non-negative")
    exact = {name: total * float(value) for name, value in ratios.items()}
    counts = {name: int(value) for name, value in exact.items()}
    remaining = total - sum(counts.values())
    order = sorted(exact, key=lambda name: (-(exact[name] - counts[name]), name))
    for name in order[:remaining]:
        counts[name] += 1
    return counts


class AssociationSampleBuilder:
    """Create validated samples without guessing labels or auxiliary targets."""

    def __init__(
        self,
        adapter: DatasetAdapter,
        history: TrajectoryHistory,
        *,
        history_length: int,
        prompt_policy: ClassPromptPolicy,
        seed: int,
    ) -> None:
        self.adapter = adapter
        self.history = history
        self.history_length = history_length
        self.prompt_policy = prompt_policy
        self.seed = seed

    def _sample(
        self,
        detection: Detection3D,
        *,
        source_id: str,
        expected_match: bool,
        auxiliary_quality: float,
        sampling_type: SamplingType,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> AssociationSample:
        candidate = AssociationCandidate(
            detection,
            expected_match=expected_match,
            auxiliary_quality=auxiliary_quality,
            candidate_source_id=source_id,
        )
        context = serialize_association_context(
            self.history,
            candidate,
            history_length=self.history_length,
            class_prompt_policy=self.prompt_policy,
            class_split=self.adapter.class_split,
        )
        return AssociationSample(
            history=self.history,
            candidate=candidate,
            semantic_prompt=context.prompt,
            target_token="Yes" if expected_match else "No",
            dataset_name=self.adapter.name,
            coordinate_frame=self.adapter.coordinate_frame,
            context=context,
            mode=AssociationMode.TRAINING,
            sampling_type=sampling_type,
            metadata=dict(metadata or {}),
        )

    def positive_exact(
        self, detection: Detection3D, *, source_id: str, auxiliary_quality: float
    ) -> AssociationSample:
        if source_id != self.history.track_id:
            raise ValueError("positive source identity must equal history identity")
        return self._sample(
            detection,
            source_id=source_id,
            expected_match=True,
            auxiliary_quality=auxiliary_quality,
            sampling_type=SamplingType.POSITIVE_EXACT,
        )

    def positive_jittered(
        self,
        detection: Detection3D,
        *,
        source_id: str,
        standard_deviation: float,
        auxiliary_quality_after_jitter: Optional[float],
    ) -> AssociationSample:
        if source_id != self.history.track_id:
            raise ValueError("positive source identity must equal history identity")
        if auxiliary_quality_after_jitter is None:
            raise ValueError("jittered positive requires explicit GT-derived auxiliary quality")
        jitter = jitter_positive_detection(
            detection, standard_deviation=standard_deviation, seed=self.seed
        )
        return self._sample(
            jitter.detection,
            source_id=source_id,
            expected_match=True,
            auxiliary_quality=auxiliary_quality_after_jitter,
            sampling_type=SamplingType.POSITIVE_JITTERED,
            metadata={"jitter": asdict(jitter)},
        )

    def negative(
        self,
        anchor: Detection3D,
        candidates: Sequence[tuple],
        *,
        strategy: str,
        auxiliary_quality: Optional[float],
        hard_k: int = 3,
        local_radius_m: Optional[float] = None,
    ) -> AssociationSample:
        if auxiliary_quality is None:
            raise ValueError("negative sample requires an explicit quality target")
        selection = select_negative_candidate(
            self.history.track_id,
            anchor,
            candidates,
            strategy=strategy,
            seed=self.seed,
            hard_k=hard_k,
            local_radius_m=local_radius_m,
        )
        return self._sample(
            selection.detection,
            source_id=selection.source_id,
            expected_match=False,
            auxiliary_quality=auxiliary_quality,
            sampling_type=selection.sampling_type,
            metadata={"selection": asdict(selection)},
        )
