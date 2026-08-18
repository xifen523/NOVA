"""Deterministic base/novel classification with an explicit real-mode gate."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Tuple

from .types import ClassMembership


class UnknownClassPolicy(str, Enum):
    AS_UNKNOWN = "as_unknown"
    REJECT = "reject"


def normalize_class_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("class name must be a non-empty string")
    return "_".join(value.strip().lower().split())


@dataclass(frozen=True)
class ClassSplit:
    """Explicit canonical Base/Novel tuples, aliases, authority, and fallback."""

    base_classes: Tuple[str, ...]
    novel_classes: Tuple[str, ...]
    aliases: Tuple[Tuple[str, str], ...] = ()
    unknown_policy: UnknownClassPolicy = UnknownClassPolicy.AS_UNKNOWN
    authoritative: bool = False
    mapping_status: str = "REQUIRES_AUTHORITATIVE_CLASS_MAP"

    def __post_init__(self) -> None:
        if not isinstance(self.base_classes, tuple) or not isinstance(self.novel_classes, tuple):
            raise TypeError("base_classes and novel_classes must be tuples")
        base = tuple(normalize_class_name(value) for value in self.base_classes)
        novel = tuple(normalize_class_name(value) for value in self.novel_classes)
        if len(base) != len(set(base)) or len(novel) != len(set(novel)):
            raise ValueError("class names must be unique after normalization")
        if set(base) & set(novel):
            raise ValueError("base and novel class sets must be disjoint")
        if not isinstance(self.aliases, tuple):
            raise TypeError("aliases must be a tuple")
        normalized_aliases = tuple(sorted(
            (normalize_class_name(alias), normalize_class_name(target))
            for alias, target in self.aliases
        ))
        alias_names = tuple(alias for alias, _ in normalized_aliases)
        if len(alias_names) != len(set(alias_names)):
            raise ValueError("aliases must be unique after normalization")
        if any(target not in set(base) | set(novel) for _, target in normalized_aliases):
            raise ValueError("every alias target must be a known canonical class")
        if not isinstance(self.unknown_policy, UnknownClassPolicy):
            raise TypeError("unknown_policy must be UnknownClassPolicy")
        allowed = {"SYNTHETIC_EXAMPLE", "REQUIRES_AUTHORITATIVE_CLASS_MAP", "AUTHORITATIVE"}
        if self.mapping_status not in allowed:
            raise ValueError("mapping_status is not allowed")
        if self.authoritative != (self.mapping_status == "AUTHORITATIVE"):
            raise ValueError("authoritative must agree with mapping_status")
        object.__setattr__(self, "base_classes", base)
        object.__setattr__(self, "novel_classes", novel)
        object.__setattr__(self, "aliases", normalized_aliases)

    @classmethod
    def from_mapping(cls, document: Mapping[str, object]) -> "ClassSplit":
        allowed = {
            "base_classes", "novel_classes", "aliases", "unknown_policy",
            "authoritative", "mapping_status",
        }
        unknown = sorted(set(document) - allowed)
        if unknown:
            raise ValueError("unknown class split fields: {0}".format(", ".join(unknown)))
        base = document.get("base_classes", ())
        novel = document.get("novel_classes", ())
        aliases = document.get("aliases", {})
        if not isinstance(base, (list, tuple)) or not isinstance(novel, (list, tuple)):
            raise TypeError("base_classes and novel_classes must be sequences")
        if any(not isinstance(value, str) for value in tuple(base) + tuple(novel)):
            raise TypeError("class names must be strings")
        if not isinstance(aliases, Mapping):
            raise TypeError("aliases must be a mapping")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in aliases.items()
        ):
            raise TypeError("alias names and targets must be strings")
        policy = UnknownClassPolicy(str(document.get("unknown_policy", "as_unknown")))
        authoritative = document.get("authoritative", False)
        status = document.get("mapping_status", "REQUIRES_AUTHORITATIVE_CLASS_MAP")
        if not isinstance(authoritative, bool) or not isinstance(status, str):
            raise TypeError("authoritative and mapping_status have invalid types")
        return cls(
            base_classes=tuple(base),
            novel_classes=tuple(novel),
            aliases=tuple(aliases.items()),
            unknown_policy=policy,
            authoritative=authoritative,
            mapping_status=status,
        )

    def classify(self, category: str) -> ClassMembership:
        canonical = self.canonicalize(category)
        if canonical in self.base_classes:
            return ClassMembership.BASE
        if canonical in self.novel_classes:
            return ClassMembership.NOVEL
        if self.unknown_policy is UnknownClassPolicy.REJECT:
            raise ValueError("unknown class: {0}".format(canonical))
        return ClassMembership.UNKNOWN

    def canonicalize(self, category: str) -> str:
        """Resolve a class alias without converting unknown classes to Novel."""

        normalized = normalize_class_name(category)
        return dict(self.aliases).get(normalized, normalized)

    def validate_for_real_mode(self) -> None:
        if not self.authoritative or self.mapping_status != "AUTHORITATIVE":
            raise ValueError("real dataset mode requires an authoritative class map")
