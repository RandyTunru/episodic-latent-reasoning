"""R-JEPA (Reasoning Joint Embedding Predictive Architecture).

Compresses verbose Chain-of-Thought reasoning into a compact, continuous latent
space. A frozen pretrained encoder extracts semantic context from the
instruction; an autoregressive predictor iteratively forecasts sequential latent
reasoning states; a lightweight router triggers a dynamic halting mechanism
once the logic converges.
"""

from typing import Optional

import torch
from torch import nn

from src.models.modules.encoder import Encoder
from src.models.r_jepa.modules.predictor import (
    CrossAttentionPredictor,
    CausalAttentionPredictor,
)


class RJEPA(nn.Module):
    """Reasoning JEPA model.

    Two-process framework:
    1. A frozen encoder provides static semantic context from the instruction.
    2. An autoregressive predictor forecasts sequential latent reasoning states
       starting from a learnable ``<start>`` token, with a router that
       dynamically halts the chain once the logic has converged.

    The target representations are produced by mean-pooling each reasoning
    step's token embeddings through the same frozen encoder.
    """

    def __init__(
        self,
        encoder_kwargs: dict,
        predictor_kwargs: dict,
        is_cross_attention: bool = True,
    ):
        """Initialise the R-JEPA model.

        Args:
            encoder_kwargs: Keyword arguments for constructing the frozen encoder.
            predictor_kwargs: Keyword arguments for constructing the predictor.
            is_cross_attention: Whether to use cross-attention in the predictor.
            predictor_kwargs: Keyword arguments forwarded to the predictor
                constructor (``encoder_dim``, ``predictor_dim``,
                ``num_heads``, ``d_ff``, ``num_layers``, ``max_seq_length``,
                ``dropout``).
            encoder_max_seq_length: Maximum token sequence length the encoder
                can handle (used to size causal-attention predictor buffers).
            is_cross_attention: Use ``CrossAttentionPredictor`` when ``True``,
                ``CausalAttentionPredictor`` when ``False``.
        """
        super(RJEPA, self).__init__()
        self.encoder = Encoder(**encoder_kwargs)

        if is_cross_attention:
            self.predictor = CrossAttentionPredictor(**predictor_kwargs)
        else:
            # Causal predictor concatenates [context, start_token, steps].
            # Its max_seq_length must cover the full window.
            predictor_kwargs = {**predictor_kwargs}
            predictor_kwargs["max_seq_length"] = (
                encoder_kwargs["max_seq_length"]
                + predictor_kwargs["max_seq_length"]
                + 1
            )
            self.predictor = CausalAttentionPredictor(**predictor_kwargs)

        # Freeze the encoder - only the predictor is trained.
        for param in self.encoder.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        return list(self.predictor.parameters())

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(
        self,
        ctx_input_ids: torch.Tensor,
        ctx_attention_mask: torch.Tensor,
        step_input_ids: torch.Tensor,
        step_attention_mask: torch.Tensor,
    ):
        """Forward pass with variable-length support via padding masks.

        Args:
            ctx_input_ids: ``(B, max_ctx_len)`` token IDs for the instruction.
            ctx_attention_mask: ``(B, max_ctx_len)`` bool, ``True`` = valid
                token, ``False`` = padding.
            step_input_ids: ``(B, max_steps, max_step_len)`` token IDs for
                each reasoning step.
            step_attention_mask: ``(B, max_steps, max_step_len)`` bool,
                ``True`` = valid token, ``False`` = padding.

        Returns:
            ``(predictions, target_repr, router_logits, step_valid_mask)``

            * **predictions**: ``(B, max_steps, encoder_dim)`` - autoregressive
              latent-state predictions (position 0 is the start-token output).
            * **target_repr**: ``(B, max_steps, encoder_dim)`` - mean-pooled
              target representations for every reasoning step.
            * **router_logits**: ``(B, max_steps)`` - raw halting scores
              (one per predicted position).
            * **step_valid_mask**: ``(B, max_steps)`` bool, ``True`` = this
              step position has a real (non-padding) target.
        """
        B = ctx_input_ids.size(0)
        S = step_input_ids.size(1)  # max_steps (padded)
        L = step_input_ids.size(2)  # max_step_len (padded)

        # ---- context encoding ----
        context_repr = self.encoder(
            ctx_input_ids, attention_mask=ctx_attention_mask,
        )  # (B, max_ctx_len, encoder_dim)

        # ---- target encoding (flatten → encode → mean-pool → reshape) ----
        flat_ids = step_input_ids.reshape(B * S, L)      # (B·S, L)
        flat_mask = step_attention_mask.reshape(B * S, L) # (B·S, L)

        with torch.no_grad():
            target_repr = self.encoder(flat_ids, attention_mask=flat_mask)
            # (B·S, L, encoder_dim)

            # Mask-weighted mean pool (exclude padding tokens).
            valid_counts = flat_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            target_repr = (
                target_repr * flat_mask.unsqueeze(-1)
            ).sum(dim=1) / valid_counts  # (B·S, encoder_dim)

            target_repr = target_repr.reshape(B, S, -1)
            # (B, max_steps, encoder_dim)

        # ---- per-sample padding masks for the predictor ----
        context_padding_mask = ~ctx_attention_mask         # True = pad
        step_valid_mask = step_attention_mask.sum(dim=-1) > 0  # (B, S)
        step_padding_mask = ~step_valid_mask               # True = pad

        # ---- autoregressive prediction ----
        predictions, router_logits = self.predictor(
            x=target_repr[:, :-1, :],                   # (B, S-1, E)
            context=context_repr,                        # (B, ctx_len, E)
            context_padding_mask=context_padding_mask,    # (B, ctx_len)
            step_padding_mask=step_padding_mask[:, :-1], # (B, S-1)
        )
        # predictions:   (B, S, E)  - S positions (start_token + S-1 steps)
        # router_logits:  (B, S)    - one score per reasoning position

        return predictions, target_repr, router_logits, step_valid_mask

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
