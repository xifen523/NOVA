"""Language-model association with explicit geometry injection."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
from torch import nn

from nova.config.schema import ModelConfig
from nova.contracts.model import (
    AssociationModelInput,
    AssociationModelOutput,
    ModelContractError,
)

from .auxiliary_head import AuxiliaryQualityHead
from .geometry_encoder import GeometryEncoder
from .tokenizer import ResolvedTokenIds


def build_language_model(
    config: ModelConfig,
    model_name_or_path: Optional[str] = None,
    *,
    local_files_only: bool = True,
    allow_download: bool = False,
    torch_dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    """Construct a standard causal language model without executing remote code."""

    config.validate(for_training=False)
    if allow_download and local_files_only:
        local_files_only = False
    if not allow_download and not local_files_only:
        raise ValueError("remote download requires explicit allow_download=True")
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path or config.base_model,
        revision=config.revision,
        local_files_only=local_files_only,
        trust_remote_code=False,
        torch_dtype=torch_dtype,
    )


def inject_geometry_embeddings(
    input_ids: torch.Tensor,
    input_embeddings: torch.Tensor,
    geometry_embeddings: torch.Tensor,
    box_counts: torch.Tensor,
    box_token_id: int,
) -> torch.Tensor:
    """Replace every box-token embedding, rejecting silent count truncation."""

    if input_ids.ndim != 2 or input_embeddings.ndim != 3:
        raise ModelContractError("input IDs/embeddings must have rank 2/3")
    if input_embeddings.shape[:2] != input_ids.shape:
        raise ModelContractError("input embedding prefix must match input_ids")
    if geometry_embeddings.ndim != 2:
        raise ModelContractError("geometry_embeddings must have [total_boxes, hidden] shape")
    if geometry_embeddings.shape[-1] != input_embeddings.shape[-1]:
        raise ModelContractError("geometry and token hidden dimensions must match")
    if box_counts.ndim != 1 or box_counts.shape[0] != input_ids.shape[0]:
        raise ModelContractError("box_counts must have [batch] shape")
    if int(box_counts.sum().item()) != geometry_embeddings.shape[0]:
        raise ModelContractError("sum(box_counts) must equal geometry rows")

    output = input_embeddings.clone()
    pointer = 0
    for batch_index, count_value in enumerate(box_counts.tolist()):
        count = int(count_value)
        positions = torch.nonzero(
            input_ids[batch_index] == box_token_id, as_tuple=False
        ).squeeze(1)
        if positions.numel() != count:
            raise ModelContractError(
                "box token count must exactly equal geometry count for every sample"
            )
        if count:
            output[batch_index, positions] = geometry_embeddings[pointer:pointer + count]
        pointer += count
    return output


def yes_probability_from_logits(
    token_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    positive_token_ids: Sequence[int],
    negative_token_ids: Sequence[int],
) -> torch.Tensor:
    """Normalize positive versus negative next-token evidence for each batch item."""

    if token_logits.ndim != 3 or attention_mask.shape != token_logits.shape[:2]:
        raise ModelContractError("logits and attention mask shapes are incompatible")
    if not positive_token_ids or not negative_token_ids:
        raise ModelContractError("positive and negative token IDs are required")
    if (attention_mask.sum(dim=1) == 0).any().item():
        raise ModelContractError("every prompt must contain at least one non-padding token")
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    last_indices = torch.where(
        attention_mask.to(dtype=torch.bool), positions.unsqueeze(0), -torch.ones_like(positions).unsqueeze(0)
    ).max(dim=1).values
    batch_indices = torch.arange(token_logits.shape[0], device=token_logits.device)
    last_logits = token_logits[batch_indices, last_indices]
    positive = torch.logsumexp(last_logits[:, list(positive_token_ids)], dim=1)
    negative = torch.logsumexp(last_logits[:, list(negative_token_ids)], dim=1)
    return torch.softmax(torch.stack((negative, positive), dim=1), dim=1)[:, 1]


class AssociationModel(nn.Module):
    """Geometry injection and output contract around an injected LM-like module."""

    def __init__(
        self,
        config: ModelConfig,
        language_hidden_dim: int,
        language_model: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        config.validate(for_training=False)
        if language_hidden_dim <= 0:
            raise ValueError("language_hidden_dim must be positive")
        if config.language_hidden_dim is not None and config.language_hidden_dim != language_hidden_dim:
            raise ValueError("explicit language hidden dimensions disagree")
        self.config = config
        self.language_model = language_model
        self.geometry_encoder = GeometryEncoder(
            input_dim=config.geometry.input_dim,
            geometry_hidden_dim=config.geometry.hidden_dim,
            language_hidden_dim=language_hidden_dim,
        )
        self.auxiliary_head = AuxiliaryQualityHead(
            hidden_dim=language_hidden_dim,
            bottleneck_dim=config.auxiliary_head.hidden_dim,
        )

    def forward(
        self,
        inputs: AssociationModelInput,
        token_ids: ResolvedTokenIds,
    ) -> AssociationModelOutput:
        if self.language_model is None:
            raise RuntimeError("AssociationModel requires a loaded language model")
        inputs.validate(geometry_input_dim=self.config.geometry.input_dim)
        token_ids.validate()
        if inputs.box_token_id != token_ids.box:
            raise ModelContractError("input and resolved box-token IDs disagree")
        if inputs.geometry is None or inputs.geometry.ndim != 2:
            raise ModelContractError("association injection currently requires flattened rank-2 geometry")

        token_embeddings = self.language_model.get_input_embeddings()(inputs.input_ids)
        geometry_embeddings = self.geometry_encoder(inputs.geometry)
        auxiliary_quality = self.auxiliary_head(geometry_embeddings)
        injection_embeddings = geometry_embeddings.to(
            device=token_embeddings.device, dtype=token_embeddings.dtype
        )
        injected = inject_geometry_embeddings(
            inputs.input_ids,
            token_embeddings,
            injection_embeddings,
            inputs.box_counts,
            inputs.box_token_id,
        )
        raw_output = self.language_model(
            inputs_embeds=injected,
            attention_mask=inputs.attention_mask,
            labels=inputs.labels,
        )
        logits = raw_output.logits
        p_yes = yes_probability_from_logits(
            logits,
            inputs.attention_mask,
            token_ids.positive,
            token_ids.negative,
        )
        losses = {}
        language_loss = getattr(raw_output, "loss", None)
        if language_loss is not None:
            losses["language"] = language_loss
        output = AssociationModelOutput(
            token_logits=logits,
            p_yes=p_yes,
            auxiliary_quality=auxiliary_quality,
            loss_components=losses,
        )
        output.validate()
        return output
