"""Typed dataset adapters and prepared-data interfaces for NOVA."""

from .base import DatasetAdapter
from .class_split import ClassSplit, UnknownClassPolicy
from .kitti import KITTIAdapter
from .nuscenes import NuScenesAdapter
from .registry import build_dataset_adapter, list_dataset_adapters, register_dataset_adapter
from .sampling import (
    JitterResult,
    SampleSelection,
    jitter_positive_detection,
    select_negative_candidate,
)
from .sample_builder import AssociationSampleBuilder, allocate_sampling_counts
from .types import (
    AssociationCandidate,
    AssociationMode,
    AssociationSample,
    ClassMembership,
    CoordinateFrame,
    Detection3D,
    FrameMetadata,
    PreparedBatch,
    SamplingType,
    SerializedAssociationContext,
    SequenceMetadata,
    TrackObservation,
    TrajectoryHistory,
)
from .v2x import V2XAdapter
from .prepared import (
    ASSOCIATION_SCHEMA,
    PreparedDataError,
    prepare_associations,
    read_association_jsonl,
    validate_association_row,
    write_jsonl,
)

__all__ = [
    "AssociationCandidate",
    "AssociationMode",
    "ASSOCIATION_SCHEMA",
    "AssociationSample",
    "AssociationSampleBuilder",
    "ClassMembership",
    "ClassSplit",
    "CoordinateFrame",
    "DatasetAdapter",
    "Detection3D",
    "FrameMetadata",
    "KITTIAdapter",
    "NuScenesAdapter",
    "PreparedBatch",
    "PreparedDataError",
    "SamplingType",
    "SerializedAssociationContext",
    "SequenceMetadata",
    "TrackObservation",
    "TrajectoryHistory",
    "UnknownClassPolicy",
    "V2XAdapter",
    "build_dataset_adapter",
    "list_dataset_adapters",
    "register_dataset_adapter",
    "prepare_associations",
    "read_association_jsonl",
    "validate_association_row",
    "write_jsonl",
    "JitterResult",
    "SampleSelection",
    "jitter_positive_detection",
    "select_negative_candidate",
    "allocate_sampling_counts",
]
