"""Dataset for chain-of-thought reasoning data with step-level annotations.

Handles datasets where each example has an "instruction" and a "response"
field.  The response typically contains chain-of-thought reasoning wrapped in
think tokens (e.g. "<think>...</think>") followed by the final answer.
This dataset extracts the CoT portion, splits it into reasoning steps, and
tokenizes each step individually for the R-JEPA target encoder.
"""

import re
from typing import Optional, List, Dict, Any

from torch.utils.data import Dataset


class ReasoningDataset(Dataset):
    """tokenizes instructions and per-step reasoning chains.

    Designed for instruction-response datasets where the response mixes a
    chain-of-thought block (delimited by think tokens) with a final answer.
    The CoT block is extracted, split into reasoning steps, and each step is
    tokenized independently.  The final answer is *not* used during R-JEPA
    pretraining - the predictor learns solely from the latent dynamics of the
    reasoning chain.

    Each sample returns:
        ctx_ids:            token IDs for the instruction
        ctx_mask:           bool mask (True = valid token)
        step_ids:           list of tensors, one per reasoning step
        step_masks:         list of bool masks, one per reasoning step
        num_steps:          number of reasoning steps
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        conversation_key: str = "messages",
        open_think_token: str = "<think>",
        close_think_token: str = "</think>",
        max_ctx_token_length: int = 512,
        max_step_token_length: int = 256,
        step_separator: Optional[str] = None,
    ):
        """
        Args:
            data: List of dicts with "instruction" and "response" fields
                (key names configurable).
            tokenizer: A HuggingFace tokenizer instance (must support
                "__call__" with "return_tensors='pt'").
            instruction_key: Key for the instruction text.
            response_key: Key for the response text (contains CoT + answer).
            open_think_token: Opening think-tag text (e.g. ``"<think>"``).
                Used to locate the start of the chain-of-thought block.
            close_think_token: Closing think-tag text (e.g. ``"</think>"``).
                Used to locate the end of the chain-of-thought block.
            max_ctx_token_length: Maximum token length for instruction.
            max_step_token_length: Maximum token length per reasoning step.
            step_separator: Regex or string to split CoT into steps.
                Defaults to "r'\\n\\n+'" (paragraph breaks) which works
                well with DeepSeek-R1-style CoT.  Set to
                "r'(?<=[.!?])\\s+'" for sentence-level splits.
        """
        self.data = data
        self.tokenizer = tokenizer

        self.conversation_key = conversation_key

        self.open_think_token = open_think_token
        self.close_think_token = close_think_token
        self.step_separator = step_separator or r"\n\n+"

        self.max_ctx_token_length = max_ctx_token_length
        self.max_step_token_length = max_step_token_length

    def __len__(self):
        return len(self.data)

    def _tokenize(self, text: str, max_length: int=None):
        """tokenize a single text, truncating to *max_length*."""
        out = self.tokenizer(
            text,
            truncation=(max_length is not None),
            max_length=max_length,
            padding=False,
            return_tensors="pt",
        )
        return out["input_ids"][0], out["attention_mask"][0].bool()

    def _extract_cot(self, response: str) -> Optional[str]:
        """Extract the chain-of-thought text from a response string.

        Looks for text between *open_think_token* and *close_think_token*
        (e.g. ``<think>…</think>``).  Uses a non-greedy regex match so that
        only the first think block is captured - multi-turn responses that
        contain multiple think blocks will only have their first block used.

        If either delimiter is missing, the entire response is returned as
        the CoT text.  This handles datasets that don't use think-tag
        delimiters at all (e.g. raw reasoning text).

        Returns:
            The extracted CoT text, or ``None`` if the response is empty.
        """
        pattern = (
            re.escape(self.open_think_token)
            + r"(.*?)"
            + re.escape(self.close_think_token)
        )
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback: no think tags - treat the entire response as CoT.
        # This handles datasets where reasoning is not explicitly delimited.
        return response.strip() or None

    def _parse_steps(self, cot_text: str) -> List[str]:
        """Split chain-of-thought text into reasoning steps.

        Filters out empty or whitespace-only steps.
        """
        raw = re.split(self.step_separator, cot_text)
        return [s.strip() for s in raw if s.strip()]

    def __getitem__(self, idx):
        example = self.data[idx]
        messages = example[self.conversation_key]

        instruction = messages[0]["content"]
        response = messages[1]["content"]

        ctx_ids, ctx_mask = self._tokenize(instruction, max_length=self.max_ctx_token_length)

        cot_text = self._extract_cot(response)
        steps = self._parse_steps(cot_text) if cot_text else []

        step_ids, step_masks = [], []
        for step in steps:
            ids, mask = self._tokenize(step, max_length=self.max_step_token_length)
            step_ids.append(ids)
            step_masks.append(mask)

        return {
            "ctx_ids": ctx_ids,
            "ctx_mask": ctx_mask,
            "step_ids": step_ids,
            "step_masks": step_masks,
            "num_steps": len(step_ids),
        }
