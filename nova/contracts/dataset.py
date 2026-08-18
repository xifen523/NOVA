"""Abstract dataset interface for optional third-party integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

from nova.data.types import AssociationSample


class DatasetAdapter(ABC):
    """Extension boundary implemented by concrete dataset integrations."""

    @abstractmethod
    def prepare_training_samples(self, *args: Any, **kwargs: Any) -> Iterable[AssociationSample]:
        raise NotImplementedError

    @abstractmethod
    def load_detections(self, *args: Any, **kwargs: Any) -> Iterable[Any]:
        raise NotImplementedError

    @abstractmethod
    def load_ground_truth(self, *args: Any, **kwargs: Any) -> Iterable[Any]:
        raise NotImplementedError

    @abstractmethod
    def serialize_history(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def build_prompt(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def format_submission(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
