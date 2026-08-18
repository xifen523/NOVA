"""Static pickle opcode inspection. This module never deserializes a pickle."""

from __future__ import annotations

import pickletools
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Tuple


class PickleInspectionError(ValueError):
    pass


@dataclass(frozen=True)
class PickleStaticReport:
    opcode_counts: Mapping[str, int]
    direct_globals: Tuple[str, ...]
    stack_global_count: int
    reduce_count: int
    persistent_id_count: int
    extension_count: int
    automated_load_approved: bool = False


def inspect_pickle_static(path: Path) -> PickleStaticReport:
    if path.is_symlink() or not path.is_file():
        raise PickleInspectionError("pickle path must be a regular file")
    counts = Counter()
    globals_found = set()
    try:
        with path.open("rb") as handle:
            for opcode, argument, _position in pickletools.genops(handle):
                counts[opcode.name] += 1
                if opcode.name == "GLOBAL":
                    globals_found.add(str(argument))
    except ValueError as exc:
        raise PickleInspectionError("invalid pickle opcode stream") from exc
    persistent = counts["PERSID"] + counts["BINPERSID"]
    extensions = counts["EXT1"] + counts["EXT2"] + counts["EXT4"]
    return PickleStaticReport(
        opcode_counts=dict(counts),
        direct_globals=tuple(sorted(globals_found)),
        stack_global_count=counts["STACK_GLOBAL"],
        reduce_count=counts["REDUCE"],
        persistent_id_count=persistent,
        extension_count=extensions,
    )
