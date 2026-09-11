"""R-JEPA (Reasoning Joint Embedding Predictive Architecture).

Compresses verbose Chain-of-Thought reasoning into a compact, continuous
latent space.  A frozen pretrained encoder extracts semantic context from
the instruction; an autoregressive predictor iteratively forecasts
sequential latent reasoning states; a lightweight router triggers a
dynamic halting mechanism once the logic converges.

Two variants:

* :class:`CausalRJEPA` - the causal predictor concatenates
  ``[context, <start>, reasoning steps]`` into a single stream.
* :class:`CrossRJEPA` - the cross-attention predictor cross-attends the
  reasoning stream to the context.

Both variants expose the same interface and accept BOTH input modes
through one branched :meth:`RJEPABase.forward`, which dispatches on the
batch dict's contents:

* **text mode** - ``ctx_input_ids``, ``ctx_attention_mask``,
  ``step_input_ids``, ``step_attention_mask`` (requires a resident
  encoder);
* **precomputed mode** - ``step_targets`` + ``num_steps`` plus the
  variant's context keys (``packed_input``/``packed_valid_mask``/
  ``ctx_len`` for causal, ``ctx_embeddings``/``ctx_attention_mask`` for
  cross) - consumed from storage, no encoder needed.

All modes return the shared
``(predictions, targets, router_logits, step_valid_mask)`` contract.
"""

from typing import Dict, Optional

import torch
from torch import nn

from src.models.r_jepa.modules.predictor import (
    CrossAttentionPredictor,
    CausalAttentionPredictor,
)


def _require(inputs: Dict[str, torch.Tensor], *keys: str) -> None:
    """Validate that a batch dict carries the keys this path needs."""
    missing = [k for k in keys if k not in inputs]
    if missing:
        raise ValueError(
            f"missing batch keys {missing} for this path (batch has {sorted(inputs)})"
        )


class RJEPABase(nn.Module):
    """Shared R-JEPA machinery: encoder handling, text path, inference.

    Not meant to be instantiated directly
    use `CausalRJEPA` or `CrossRJEPA` (or the `RJEPA` factory).

    The encoder is optional: precomputed-only training never touches it,
    but the text path and `infer` require one.  The target
    representations are produced by pooling each reasoning step's token
    embeddings through the frozen encoder; the pooling mode is set by
    `encoder_pooling` and should match the encoder family.
    """

    def __init__(
        self,
        predictor: nn.Module,
        encoder_max_seq_length: int,
        encoder: Optional[nn.Module] = None,
        encoder_pooling: Optional[str] = "mean",
    ):
        """Args:
            predictor: Pre-built predictor module (the variant classes
                construct their own).
            encoder: A pre-built encoder module.  Must accept
                ``(input_ids, attention_mask) -> (B, seq_len, d_model)``,
                or ``None`` for precomputed-only training.
            encoder_max_seq_length: Maximum token sequence length the
                encoder can handle.  Required; sizes the causal
                predictor's position buffers.
            encoder_pooling: Pooling method for the per-step target
                representations: ``"mean"`` (mask-weighted mean of valid
                tokens), ``"cls"`` (first token), or ``"eos"`` (last
                valid token).
        """
        super(RJEPABase, self).__init__()

        if encoder_pooling and encoder_pooling not in ["mean", "cls", "eos"]:
            raise ValueError(
                f"Invalid pooling method '{encoder_pooling}'. Must be one of "
                "'mean', 'cls', or 'eos'."
            )
        self.encoder_pooling = encoder_pooling

        if encoder_max_seq_length is None:
            raise ValueError(
                "encoder_max_seq_length is required (sizes the causal "
                "predictor's position buffers)"
            )
        self.enc_max_len = encoder_max_seq_length
        self.encoder = encoder
        self.predictor = predictor

        # Freeze the encoder - only the predictor is trained.
        if self.encoder is not None:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def trainable_parameters(self):
        return list(self.predictor.parameters())

    def train(self, mode: bool = True):
        super().train(mode)
        if self.encoder is not None:
            self.encoder.eval()
        return self

    def forward(self, inputs: Dict[str, torch.Tensor]):
        """Branch on the batch dict's contents.

        The trainer passes the collated batch straight through:

        * precomputed mode when `step_targets` is present - consumed
          from storage, no encoder (variant-specific implementation in
          `_forward_precomputed`).
        * text mode otherwise - token ids/masks, encoded online.

        Returns:
            `(predictions, targets, router_logits, step_valid_mask)`
            where predictions/targets are ``(B, max_steps, E)`` and the
            masks are ``(B, max_steps)`` bool (for loss computation).
        """
        if "step_targets" in inputs:
            return self._forward_precomputed(inputs)
        return self._forward_text(inputs)

    def _forward_text(self, inputs: Dict[str, torch.Tensor]):
        """Text path: encode ctx + step tokens online, then predict."""
        _require(
            inputs,
            "ctx_input_ids", "ctx_attention_mask",
            "step_input_ids", "step_attention_mask",
        )
        if self.encoder is None:
            raise RuntimeError(
                "the text path requires an encoder; this model was built "
                "precomputed-only (encoder=None)"
            )
        ctx_input_ids = inputs["ctx_input_ids"]
        ctx_attention_mask = inputs["ctx_attention_mask"]
        step_input_ids = inputs["step_input_ids"]
        step_attention_mask = inputs["step_attention_mask"]

        B = ctx_input_ids.size(0)
        S = step_input_ids.size(1)  # max_steps (padded)
        L = step_input_ids.size(2)  # max_step_len (padded)

        # context encoding
        context_repr = self.encoder(
            ctx_input_ids, attention_mask=ctx_attention_mask,
        )  # (B, max_ctx_len, encoder_dim)

        # target encoding (flatten -> encode -> pool -> reshape)
        flat_ids = step_input_ids.reshape(B * S, L)       # (B·S, L)
        flat_mask = step_attention_mask.reshape(B * S, L)  # (B·S, L)

        with torch.no_grad():
            target_repr = self.encoder(flat_ids, attention_mask=flat_mask)
            # (B·S, L, encoder_dim)

            # Pool each step's token embeddings into a single vector.
            target_repr = self._pool(target_repr, attention_mask=flat_mask)
            # (B·S, encoder_dim)

            target_repr = target_repr.reshape(B, S, -1)
            # (B, max_steps, encoder_dim)

        # per-sample padding masks for the predictor
        context_padding_mask = ~ctx_attention_mask          # True = pad
        step_valid_mask = step_attention_mask.sum(dim=-1) > 0  # (B, S)
        step_padding_mask = ~step_valid_mask                # True = pad

        # autoregressive prediction
        predictions, router_logits = self.predictor(
            x=target_repr[:, :-1, :],                   # (B, S-1, E)
            context=context_repr,                        # (B, ctx_len, E)
            context_padding_mask=context_padding_mask,    # (B, ctx_len)
            step_padding_mask=step_padding_mask[:, :-1],  # (B, S-1)
        )
        # predictions:   (B, S, E) - S positions (start_token + S-1 steps)
        # router_logits:  (B, S)   - one score per reasoning position

        return predictions, target_repr, router_logits, step_valid_mask

    def _forward_precomputed(self, inputs: Dict[str, torch.Tensor]):
        """Precomputed path - variant-specific (no encoder)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement the precomputed path"
        )

    @staticmethod
    def _step_valid(num_steps: torch.Tensor, S: int, device: torch.device):
        """(B, S) bool mask: True = this step position has a real target."""
        return torch.arange(S, device=device).unsqueeze(0) < num_steps.unsqueeze(1)

    # ------------------------------------------------------------------
    # Pooling + inference (shared)
    # ------------------------------------------------------------------

    def _pool(self, encoder_output: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Pool the encoder output according to ``self.encoder_pooling``.

        Args:
            encoder_output: ``(B, seq_len, d_model)`` tensor from the encoder.
            attention_mask: ``(B, seq_len)`` bool mask, ``True`` = valid token.
                Required for 'mean' and 'eos' pooling; ignored by 'cls'.

        Returns:
            Pooled representation: ``(B, d_model)``
        """
        if self.encoder_pooling == 'mean':
            if attention_mask is None:
                raise ValueError("attention_mask is required for mean pooling")
            valid_counts = attention_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            pooled_output = (encoder_output * attention_mask.unsqueeze(-1)).sum(dim=1) / valid_counts
        elif self.encoder_pooling == 'cls':
            pooled_output = encoder_output[:, 0, :]  # First token (CLS/BOS)
        elif self.encoder_pooling == 'eos':
            if attention_mask is None:
                raise ValueError("attention_mask is required for eos pooling")
            # Last valid (non-padding) token; clamp so fully-padded rows
            # index position 0 instead of -1.
            valid_counts = attention_mask.sum(dim=-1).clamp(min=1)
            pooled_output = encoder_output[
                torch.arange(encoder_output.size(0)), valid_counts - 1
            ]
        else:
            raise ValueError(f"Unsupported pooling method: {self.encoder_pooling}")

        return pooled_output

    @torch.no_grad()
    def infer(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_steps: int = 256,
    ):
        """Run iterative autoregressive inference.

        Args:
            input_ids: ``(1, seq_len)`` token IDs (batch size must be 1).
            attention_mask: ``(1, seq_len)`` bool mask, ``True`` = valid.
            max_steps: Maximum number of reasoning steps to generate.

        Returns:
            ``(1, num_steps, encoder_dim)`` tensor of latent reasoning states.
        """
        if self.encoder is None:
            raise RuntimeError(
                "infer() requires an encoder; this model was built "
                "precomputed-only (encoder=None)"
            )
        # Context branch.
        context_repr = self.encoder(
            input_ids, attention_mask=attention_mask,
        )  # (1, seq_len, encoder_dim)

        # Accumulate reasoning states iteratively.
        reasoning_steps = torch.empty(
            (input_ids.size(0), 0, context_repr.size(-1)),
            device=input_ids.device,
        )

        for _ in range(max_steps):
            predictions, router_logits = self.predictor(
                x=reasoning_steps,
                context=context_repr,
            )
            # predictions: (1, cur_steps + 1, E) - includes start_token output

            # Append the newest prediction.
            reasoning_steps = torch.cat(
                [reasoning_steps, predictions[:, -1:, :]], dim=1
            )

            # Halt if the router signals convergence.
            if router_logits[:, -1] < 0:
                break

        return reasoning_steps


class CausalRJEPA(RJEPABase):
    """Causal variant: one stream ``[context, <start>, reasoning steps]``.

    ``predictor_kwargs["max_seq_length"]`` is the reasoning window; the
    causal predictor's full window (context + start token + steps) is
    sized internally by adding ``encoder_max_seq_length + 1``.

    Precomputed mode consumes the packed batch from
    ``pad_collate_precomputed_causal``: ``packed_input``,
    ``packed_valid_mask``, ``ctx_len``, ``step_targets``, ``num_steps``.
    """

    def __init__(
        self,
        predictor_kwargs: dict,
        encoder_max_seq_length: int,
        encoder: Optional[nn.Module] = None,
        encoder_pooling: str = "mean",
    ):
        predictor_kwargs = dict(predictor_kwargs)
        predictor_kwargs["max_seq_length"] = (
            encoder_max_seq_length
            + predictor_kwargs["max_seq_length"]
            + 1  # +1 for the learnable start_token
        )
        predictor = CausalAttentionPredictor(**predictor_kwargs)
        super().__init__(
            predictor,
            encoder_max_seq_length=encoder_max_seq_length,
            encoder=encoder,
            encoder_pooling=encoder_pooling,
        )

    def _forward_precomputed(self, inputs: Dict[str, torch.Tensor]):
        """Precomputed path: packed stream + stored targets (no encoder)."""
        _require(
            inputs,
            "packed_input", "packed_valid_mask", "ctx_len",
            "step_targets", "num_steps",
        )
        step_targets = inputs["step_targets"]
        S = step_targets.size(1)
        step_valid_mask = self._step_valid(
            inputs["num_steps"], S, step_targets.device,
        )

        predictions, router_logits = self.predictor.forward_packed(
            inputs["packed_input"],
            inputs["packed_valid_mask"],
            inputs["ctx_len"],
            S_out=S,
        )
        return predictions, step_targets, router_logits, step_valid_mask


class CrossRJEPA(RJEPABase):
    """Cross-attention variant: reasoning stream cross-attends to context.

    Precomputed mode consumes the per-axis batch from
    ``pad_collate_precomputed_cross``: ``ctx_embeddings``,
    ``ctx_attention_mask``, ``step_targets``, ``num_steps``.
    """

    def __init__(
        self,
        predictor_kwargs: dict,
        encoder_max_seq_length: int,
        encoder: Optional[nn.Module] = None,
        encoder_pooling: str = "mean",
    ):
        predictor = CrossAttentionPredictor(**dict(predictor_kwargs))
        super().__init__(
            predictor,
            encoder_max_seq_length=encoder_max_seq_length,
            encoder=encoder,
            encoder_pooling=encoder_pooling,
        )

    def _forward_precomputed(self, inputs: Dict[str, torch.Tensor]):
        """Precomputed path: per-axis tensors + stored targets (no encoder)."""
        _require(
            inputs,
            "ctx_embeddings", "ctx_attention_mask",
            "step_targets", "num_steps",
        )
        step_targets = inputs["step_targets"]
        S = step_targets.size(1)
        step_valid_mask = self._step_valid(
            inputs["num_steps"], S, step_targets.device,
        )

        predictions, router_logits = self.predictor(
            x=step_targets[:, :-1, :],
            context=inputs["ctx_embeddings"],
            context_padding_mask=~inputs["ctx_attention_mask"],
            step_padding_mask=~step_valid_mask[:, :-1],
        )
        return predictions, step_targets, router_logits, step_valid_mask


def RJEPA(
    predictor_kwargs: dict,
    encoder_max_seq_length: int,
    encoder: Optional[nn.Module] = None,
    encoder_pooling: Optional[str] = "mean",
    cross_attention: bool = False,
) -> RJEPABase:
    """Compatibility constructor - returns the matching variant class."""
    if cross_attention:
        return CrossRJEPA(
            predictor_kwargs,
            encoder=encoder,
            encoder_max_seq_length=encoder_max_seq_length,
            encoder_pooling=encoder_pooling,
        )
    return CausalRJEPA(
        predictor_kwargs,
        encoder=encoder,
        encoder_max_seq_length=encoder_max_seq_length,
        encoder_pooling=encoder_pooling,
    )
