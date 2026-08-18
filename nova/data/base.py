"""Abstract in-memory adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Optional, Sequence

import torch

from nova.preprocessing.association import ClassPromptPolicy, serialize_association_context
from nova.preprocessing.common import target_from_expected
from nova.preprocessing.geometry import build_geometry_9d, build_geometry_batch
from nova.preprocessing.validation import require_string

from .class_split import ClassSplit
from .types import (
    AssociationCandidate,
    AssociationMode,
    AssociationSample,
    CoordinateFrame,
    Detection3D,
    PreparedBatch,
    TrackObservation,
    TrajectoryHistory,
    SamplingType,
)


class DatasetAdapter(ABC):
    """Convert validated, caller-owned mappings into immutable NOVA contracts."""

    name: str
    coordinate_frame: CoordinateFrame

    def __init__(self, class_split: ClassSplit, real_mode: bool = False) -> None:
        if not isinstance(class_split, ClassSplit):
            raise TypeError("class_split must be ClassSplit")
        if not isinstance(real_mode, bool):
            raise TypeError("real_mode must be bool")
        if real_mode:
            class_split.validate_for_real_mode()
        self.class_split = class_split
        self.real_mode = real_mode

    @abstractmethod
    def build_detection(self, record: Mapping[str, object]) -> Detection3D:
        """Build one detection without mutating the input mapping."""

    def validate_record(self, record: Mapping[str, object]) -> None:
        self.build_detection(record)

    def build_history(
        self, track_id: str, records: Sequence[Mapping[str, object]]
    ) -> TrajectoryHistory:
        if not isinstance(track_id, str) or not track_id.strip():
            raise ValueError("track_id must be a non-empty string")
        if not isinstance(records, (list, tuple)):
            raise TypeError("records must be a list or tuple")
        observations = []
        for record in records:
            record_track_id = require_string(record, "track_id")
            if record_track_id != track_id:
                raise ValueError("record track_id does not match requested history")
            observations.append(
                TrackObservation(track_id=track_id, detection=self.build_detection(record))
            )
        return TrajectoryHistory(track_id=track_id, observations=tuple(observations))

    def build_association_sample(
        self,
        history: TrajectoryHistory,
        candidate_record: Mapping[str, object],
        expected_match: Optional[bool] = None,
        auxiliary_quality: Optional[float] = None,
        *,
        candidate_source_id: Optional[str] = None,
        mode: AssociationMode = AssociationMode.INFERENCE,
        sampling_type: Optional[SamplingType] = None,
        history_length: Optional[int] = None,
        class_prompt_policy: Optional[ClassPromptPolicy] = None,
    ) -> AssociationSample:
        if not isinstance(history, TrajectoryHistory):
            raise TypeError("history must be TrajectoryHistory")
        if not isinstance(mode, AssociationMode):
            raise TypeError("mode must be AssociationMode")
        if candidate_source_id is None:
            raw_source = candidate_record.get("candidate_source_id")
            candidate_source_id = raw_source if isinstance(raw_source, str) else None
        if mode is AssociationMode.TRAINING:
            if expected_match is None:
                raise ValueError("training samples require explicit expected_match")
            if candidate_source_id is None:
                raise ValueError("training samples require candidate_source_id")
            if sampling_type is None:
                raise ValueError("training samples require sampling_type")
        elif mode is AssociationMode.INFERENCE and expected_match is not None:
            raise ValueError("inference samples must not carry expected_match")
        contract_fields = {
            "candidate_source_id", "expected_match", "auxiliary_quality", "sampling_type",
            "metadata", "ground_truth_track_id",
        }
        detection_record = {
            key: value for key, value in candidate_record.items() if key not in contract_fields
        }
        detection = self.build_detection(detection_record)
        candidate = AssociationCandidate(
            detection=detection,
            expected_match=expected_match,
            auxiliary_quality=auxiliary_quality,
            candidate_source_id=candidate_source_id,
        )
        target = target_from_expected(expected_match)
        if history_length is None:
            history_length = len(history.observations)
        context = serialize_association_context(
            history,
            candidate,
            history_length=history_length,
            class_prompt_policy=class_prompt_policy or ClassPromptPolicy(),
            class_split=self.class_split,
        )
        sample = AssociationSample(
            history=history,
            candidate=candidate,
            semantic_prompt=context.prompt,
            target_token=target,
            dataset_name=self.name,
            coordinate_frame=self.coordinate_frame,
            context=context,
            mode=mode,
            sampling_type=sampling_type,
        )
        return sample

    def build_geometry_features(self, sample: AssociationSample) -> torch.Tensor:
        if not isinstance(sample, AssociationSample):
            raise TypeError("sample must be AssociationSample")
        if sample.context is None:
            return build_geometry_9d(sample.candidate.detection).unsqueeze(0)
        return torch.tensor(sample.context.geometry_features, dtype=torch.float32)

    def build_prepared_batch(
        self, samples: Sequence[AssociationSample]
    ) -> PreparedBatch:
        sample_tuple = tuple(samples)
        return PreparedBatch(
            samples=sample_tuple,
            geometry=build_geometry_batch(sample_tuple),
            box_counts=tuple(
                sample.context.box_count if sample.context is not None else 1
                for sample in sample_tuple
            ),
        )
