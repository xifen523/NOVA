"""Known paper/local metric conflicts that cannot satisfy release acceptance."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Mapping, Tuple

import yaml


@dataclass(frozen=True)
class KnownConflict:
    conflict_id: str
    status: str
    release_acceptance_target: bool
    paper_metrics: Mapping[str, Decimal]
    verified_local_metrics: Mapping[str, Decimal]
    absolute_differences: Mapping[str, Decimal]
    material_conflict: bool
    paper_reference: str
    verification_note: str

    def __post_init__(self) -> None:
        if self.status != "KNOWN_CONFLICT" or self.release_acceptance_target:
            raise ValueError("known conflict can never be a release acceptance target")
        if not self.paper_reference or not self.verification_note:
            raise ValueError("known conflict provenance is incomplete")

    def passes_release_acceptance(self) -> bool:
        return False


def load_known_conflicts(path: Path) -> Tuple[KnownConflict, ...]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or set(document) != {"conflicts"}:
        raise ValueError("conflict root must contain only conflicts")
    conflicts = []
    for entry in document["conflicts"]:
        convert = lambda values: {name: Decimal(str(value)) for name, value in values.items()}
        conflicts.append(KnownConflict(
            conflict_id=entry["conflict_id"], status=entry["status"],
            release_acceptance_target=entry["release_acceptance_target"],
            paper_metrics=convert(entry["paper_metrics"]),
            verified_local_metrics=convert(entry["verified_local_metrics"]),
            absolute_differences=convert(entry["absolute_differences"]),
            material_conflict=entry["material_conflict"],
            paper_reference=entry["paper_reference"],
            verification_note=entry["verification_note"],
        ))
    return tuple(conflicts)
