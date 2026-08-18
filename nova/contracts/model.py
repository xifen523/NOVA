"""Model tensor contracts independent of a real language model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


class ModelContractError(ValueError):
    """Raised when a model boundary invariant is violated."""


@dataclass
class AssociationModelInput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    geometry: Optional[torch.Tensor]
    box_counts: torch.Tensor
    box_token_id: int
    labels: Optional[torch.Tensor] = None
    auxiliary_target: Optional[torch.Tensor] = None

    def validate(self, geometry_input_dim: int = 9) -> None:
        if self.geometry is None:
            raise ModelContractError("geometry is required")
        if self.input_ids.ndim != 2 or self.attention_mask.shape != self.input_ids.shape:
            raise ModelContractError("input_ids and attention_mask must share [batch, sequence] shape")
        if self.geometry.ndim not in (2, 3):
            raise ModelContractError("geometry must have rank 2 or 3")
        if self.geometry.shape[-1] != geometry_input_dim:
            raise ModelContractError(
                "geometry last dimension must be {0}".format(geometry_input_dim)
            )
        if not torch.isfinite(self.geometry).all().item():
            raise ModelContractError("geometry must contain only finite values")
        if self.box_counts.ndim != 1 or self.box_counts.shape[0] != self.input_ids.shape[0]:
            raise ModelContractError("box_counts must have [batch] shape")
        if self.box_counts.dtype == torch.bool or torch.is_floating_point(self.box_counts):
            raise ModelContractError("box_counts must use an integer dtype")
        if (self.box_counts < 0).any().item():
            raise ModelContractError("box_counts cannot be negative")
        if self.geometry.ndim == 2:
            if int(self.box_counts.sum().item()) != self.geometry.shape[0]:
                raise ModelContractError("sum(box_counts) must equal flattened geometry rows")
        elif self.geometry.shape[:2] != (
            self.input_ids.shape[0],
            int(self.box_counts.max().item()) if self.box_counts.numel() else 0,
        ):
            raise ModelContractError(
                "rank-3 geometry must be a dense [batch, max_boxes, 9] tensor"
            )



@dataclass
class AssociationModelOutput:
    token_logits: torch.Tensor
    p_yes: torch.Tensor
    auxiliary_quality: Optional[torch.Tensor] = None
    loss_components: Dict[str, torch.Tensor] = field(default_factory=dict)

    def validate(self) -> None:
        if self.token_logits.ndim != 3:
            raise ModelContractError("token_logits must have [batch, sequence, vocabulary] shape")
        if self.p_yes.ndim != 1 or self.p_yes.shape[0] != self.token_logits.shape[0]:
            raise ModelContractError("p_yes must have [batch] shape")
        if not torch.isfinite(self.p_yes).all().item():
            raise ModelContractError("p_yes must be finite")
        if ((self.p_yes < 0) | (self.p_yes > 1)).any().item():
            raise ModelContractError("p_yes must be in [0, 1]")
