"""Generative model utilities.

Currently provides the MPNNTokenizer for converting amino acid sequences
to ProteinMPNN token indices.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedTokenizerBase
from typing import Protocol, runtime_checkable, Optional, Callable, Dict, List, Any
from contextlib import contextmanager
from abc import ABC, abstractmethod
from proteingen.probability_model import ProbabilityModel


TransitionFunc = Callable[
    torch.LongTensor, torch.FloatTensor
]  # takes in sequence outputs log_probs


class TransitionModel(ProbabilityModel):
    """Wraps a model + logit formatter into a ready-to-use probability model.

    Pass in any ``nn.Module`` whose forward returns logits, a tokenizer, and a
    ``LogitFormatter`` (e.g. ``MaskedModelLogitFormatter`` for masked diffusion,
    ``PassThroughLogitFormatter`` for uniform noise). ``TransitionModel`` handles
    the rest: forward runs the wrapped model, applies logit formatting, and
    ``get_log_probs`` (inherited from ``ProbabilityModel``) adds temperature-scaled
    log_softmax.

    For structure-conditioned models, subclass and override
    ``preprocess_observations`` and ``collate_observations``, then use
    ``set_condition_()`` / ``conditioned_on()`` to cache structure tensors.

    Example::

        esmc = ESMC.from_pretrained("esmc_300m")
        tokenizer = EsmSequenceTokenizer()
        formatter = MaskedModelLogitFormatter(tokenizer)
        model = TransitionModel(esmc, tokenizer, formatter)

        # unconditional
        log_probs = model.get_log_probs(seq_SP)

        # with temperature
        with model.with_temp(0.5):
            log_probs = model.get_log_probs(seq_SP)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        logit_formatter: LogitFormatter,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.logit_formatter = logit_formatter

    def forward(self, seq_SP: torch.LongTensor, **kwargs) -> torch.FloatTensor:
        raw_forward_output = self.model(seq_SP, **kwargs)
        return raw_forward_output

    def format_raw_to_logits(
        self, raw_forward_output: torch.FloatTensor, seq_SP: torch.LongTensor, **kwargs
    ) -> torch.FloatTensor:
        """Default: model produces outputs, they might just not be masked properly. Most common function to override."""
        logits_SPT = raw_forward_output
        logits_SPT = self.logit_formatter(logits_SPT, seq_SP)
        return logits_SPT

    def get_log_probs_from_string(
        self, sequences: list[str]
    ):  # TODO[pi] check all the type hints in the src folder
        tokenized = self.tokenizer(sequences, padding=True, return_tensors="pt")
        seq_SP = tokenized["input_ids"].to(device=self.device, dtype=torch.long)
        return self.get_log_probs(seq_SP)

    # ── LoRA ─────────────────────────────────────────────────────────────

    @property
    def has_lora(self) -> bool:
        """Whether a PEFT LoRA adapter is currently applied to self.model."""
        try:
            from peft import PeftModel

            return isinstance(self.model, PeftModel)
        except ImportError:
            return False

    def lora_target_modules(self) -> dict[str, tuple[int, int, int]]:
        """Discover Linear modules in the wrapped model — potential LoRA targets.

        Returns dict mapping name patterns to ``(in_features, out_features, count)``.
        Block-level numeric indices are collapsed to ``*`` for readability.

        Example for ESMC-300m::

            {
                'transformer.blocks.*.attn.layernorm_qkv.1': (960, 2880, 30),
                'transformer.blocks.*.attn.out_proj':         (960, 960,  30),
                'transformer.blocks.*.ffn.1':                 (960, 5120, 30),
                'transformer.blocks.*.ffn.3':                 (2560, 960, 30),
                'sequence_head.0':                            (960, 960,  1),
                'sequence_head.2':                            (960, 64,   1),
            }
        """
        groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.Linear):
                # Collapse .N. block indices to .*. but keep trailing .N
                pattern = re.sub(r"(?<=\.)\d+(?=\.)", "*", name)
                groups[pattern].append((mod.in_features, mod.out_features))
        return {
            pattern: (dims[0][0], dims[0][1], len(dims))
            for pattern, dims in groups.items()
        }

    def apply_lora(
        self,
        target_modules: list[str] | None = None,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        bias: str = "none",
        **kwargs,
    ) -> None:
        """Apply PEFT LoRA adapters to the wrapped model.

        Freezes base model parameters and injects trainable low-rank adapters
        into the targeted Linear layers. After this call, only LoRA parameters
        in ``self.model`` have ``requires_grad=True``.

        Args:
            target_modules: Which Linear layers to adapt (matched by name
                substring). If ``None``, targets all Linear layers in the model.
                Use :meth:`lora_target_modules` to discover available targets.
            r: LoRA rank (number of low-rank dimensions).
            lora_alpha: LoRA scaling factor.
            lora_dropout: Dropout probability on LoRA layers.
            bias: Bias training mode — ``"none"``, ``"all"``, or ``"lora_only"``.
            **kwargs: Extra arguments passed to ``peft.LoraConfig``.
        """
        from peft import get_peft_model, LoraConfig

        if target_modules is None:
            target_modules = [
                name
                for name, mod in self.model.named_modules()
                if isinstance(mod, nn.Linear)
            ]
            assert len(target_modules) > 0, "No Linear layers found in model"

        config = LoraConfig(
            target_modules=target_modules,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            **kwargs,
        )
        self.model = get_peft_model(self.model, config)

    def save_lora(self, path: str | Path) -> None:
        """Save only the LoRA adapter weights and config."""
        assert self.has_lora, "No LoRA adapter to save"
        self.model.save_pretrained(str(path))

    def load_lora(self, path: str | Path) -> None:
        """Load a saved LoRA adapter onto the base model."""
        from peft import PeftModel

        assert not self.has_lora, (
            "Model already has a LoRA adapter. Use a fresh base model."
        )
        self.model = PeftModel.from_pretrained(self.model, str(path))

    # ── Checkpointing ────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save model to a directory. Includes LoRA adapter if applied."""
        super().save(path)
        if self.has_lora:
            self.model.save_pretrained(str(Path(path) / "lora_adapter"))

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "TransitionModel":
        """Load model from a directory. Loads LoRA adapter if present."""
        obj = super().from_checkpoint(path)
        lora_path = Path(path) / "lora_adapter"
        if lora_path.exists():
            obj.load_lora(lora_path)
        return obj


class TransitionModelWithEmbedding(TransitionModel, ABC):
    """TransitionModel that exposes a differentiable embedding step.

    Subclasses implement two methods that split the model's forward pass:

    - ``differentiable_embedding(ohe_seq_SPT)`` — OHE float input → deep
      embeddings (after transformer / encoder body).
    - ``embedding_to_outputs(embedding_SPD)`` — deep embeddings → raw outputs

    This class provides concrete ``forward`` and ``format_raw_to_logits``
    that compose these two steps, create a differentiable OHE from token IDs
    (so gradients flow through the embedding step for TAG), and apply the
    logit formatter.

    Subclasses must set ``EMB_DIM`` (int) for downstream use (e.g. LinearProbe).
    """

    EMB_DIM: int  # subclasses must set this

    @abstractmethod
    def differentiable_embedding(
        self, ohe_seq_SPT: torch.FloatTensor
    ) -> torch.FloatTensor:
        """OHE (or soft distribution) over tokens → deep embeddings (S, P, D).

        Typically: ``ohe @ embed.weight`` → transformer body.
        """
        ...

    @abstractmethod
    def embedding_to_outputs(self, embedding_SPD: torch.FloatTensor) -> Any:
        """Deep embeddings → regular raw model outputs. IMPORTANT, the output of this function should be of the same type as the forward function!"""
        ...

    def embed(self, seq_SP: torch.LongTensor) -> torch.FloatTensor:
        """Token IDs → deep embeddings (S, P, D)."""
        ohe_seq_SPT = F.one_hot(seq_SP, num_classes=self.tokenizer.vocab_size).float()
        return self.differentiable_embedding(ohe_seq_SPT)


@runtime_checkable
class LogitFormatter(Protocol):
    """Constrains model output logits based on input token identities.

    Applied before log_softmax to enforce valid output distributions
    per input token (e.g. special tokens predict themselves, mask tokens
    predict only non-special tokens).

    Implementations intended for use as model submodules should inherit
    from nn.Module and use register_buffer for device tracking. When
    inheriting from both nn.Module and LogitFormatter, nn.Module must
    come first in the MRO (e.g. ``class Foo(nn.Module, LogitFormatter)``)
    so that nn.Module.__call__ (which dispatches to forward) is resolved
    before Protocol.__call__.

    Must return a FloatTensor so that the softmax doesn't have normalization
    issues due to a lack of precision.

    Design note — possible implementation approaches:
        - **Reference implementation (MaskedModelLogitFomatter)** uses a precomputed
        dense additive mask matrix indexed by input token ids. The translation matrix
        is built once at init and reused every forward pass, fully vectorized with
        no branching. Alternative approaches include:
        - **In-place scatter**: no precomputed matrix; loop over positions at forward
          time and write -inf into invalid outputs. Simple but slow.
        - **Boolean mask + masked_fill**: store a boolean matrix (1 bit vs 32 bits
          per entry), index it the same way, then ``logits.masked_fill(~mask, -inf)``.
          Saves memory at the cost of an extra op.
        - **Sparse allowlist**: store a dict mapping each token id to a LongTensor
          of valid output indices. More natural for huge vocabularies where the
          valid set per token is tiny.
        - **Categorical branching**: classify each input token as mask/special/regular
          and apply a different rule per type. No matrix, but introduces branching.
        - **Post-softmax renormalization**: run softmax normally, zero out invalid
          probs, renormalize. Changes the gradient landscape vs. additive masking.
        - **Loss-side only**: don't constrain logits at all; mask the loss instead
          and trust the model learns the constraints. No guarantees at inference.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        output_dim: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.output_dim = output_dim

    def __call__(
        self, logits: torch.Tensor, input_ids: torch.LongTensor
    ) -> torch.FloatTensor: ...


class PassThroughLogitFormatter(LogitFormatter):
    def __init__(self):
        pass

    def __call__(
        self, logits: torch.Tensor, input_ids: torch.LongTensor
    ) -> torch.FloatTensor:
        return logits.float()


class MaskedModelLogitFormatter(nn.Module, LogitFormatter):
    """Enforces output constraints for masked language models via additive masking.

    Builds a static mask matrix of shape (Ti, To) that defines which output tokens
    are valid for each input token. In forward, input token ids directly index into
    this matrix to select the per-position mask, which is then added to the raw
    logits before log_softmax.

    Constraints:
        - Special tokens (CLS, EOS, PAD, etc.) can only predict themselves.
        - The mask token can predict any non-special token (but not itself).
        - All other tokens (standard vocabulary) predict only themselves.

    The mask matrix contains 0.0 for valid outputs and -inf for invalid outputs,
    so adding it to logits zeros out invalid positions after softmax.

    output_dim may exceed vocab_size when model designers pad the output space
    for memory alignment (e.g. ESM's 33-token vocab mapped to 64-dim output).
    Extra columns beyond vocab_size are valid mask outputs (not special tokens).

    Tensor index conventions:
        Ti: input token index — rows of the mask matrix, size = vocab_size
        To: output token index — columns of the mask matrix, size = output_dim
        S:  batch (sequence) index
        P:  position index within a sequence
        T:  token/vocab dimension in logits (same axis as To)
    """

    # Standard amino acid one-letter codes (canonical 20).
    STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")

    # TODO[pi] move the init to be a "from_hf_tokenizer" method and make the actual init general like we discussed
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        output_dim: Optional[int] = None,
        canonical_only: bool = True,
    ):

        # nn.Module.__init__(self)
        super().__init__()
        self.tokenizer = tokenizer
        if output_dim is None:
            self.output_dim = self.tokenizer.vocab_size
        else:
            self.output_dim = output_dim
        assert self.output_dim >= self.tokenizer.vocab_size, (
            "Outputs can't include all tokens!! Output dim is less than the tokenizer vocab size"
        )  # TODO: add support for if output tokens are a subset of the tokenizer--maybe when create the constructor for non HF-tokenizer

        # Construct valid output mask: 0.0 = pass through, -inf = block
        PASS_THROUGH, BLOCK_OUT = 0.0, float("-inf")

        valid_output_mask_TiTo = torch.full(
            (self.tokenizer.vocab_size, self.output_dim), BLOCK_OUT, dtype=torch.float32
        )

        # All tokens map to themselves
        for idx in self.tokenizer.vocab.values():
            valid_output_mask_TiTo[idx, idx] = PASS_THROUGH
        # mask_token_idx = self.tokenizer.vocab[mask_token]
        mask_token_idx = self.tokenizer.mask_token_id

        # Except for mask which maps to any non-special token
        valid_output_mask_TiTo[mask_token_idx, mask_token_idx] = BLOCK_OUT
        special_ids = set(self.tokenizer.added_tokens_decoder.keys())
        if canonical_only:
            # Only allow mask → standard amino acid transitions
            idx_to_tok = {idx: tok for tok, idx in self.tokenizer.vocab.items()}
            canonical_ids = {
                idx
                for idx in range(self.tokenizer.vocab_size)
                if idx not in special_ids
                and idx_to_tok.get(idx, "") in self.STANDARD_AAS
            }
            # Fall back to all non-special if vocab has no standard AAs
            # (e.g. BERT word-piece tokenizer)
            valid_mask_outputs = canonical_ids if canonical_ids else (
                set(range(self.tokenizer.vocab_size)) - special_ids
            )
        else:
            valid_mask_outputs = set(range(self.tokenizer.vocab_size)) - special_ids
        for idx in valid_mask_outputs:
            valid_output_mask_TiTo[mask_token_idx, idx] = PASS_THROUGH

        self.register_buffer("valid_output_mask_TiTo", valid_output_mask_TiTo)

    def forward(self, logits_SPT: torch.Tensor, seq_SP: torch.LongTensor):
        """Apply per-position output constraints to raw logits.

        Indexes the mask matrix by input token ids to select the constraint
        row for each position, then adds it to the logits. Positions with
        special tokens will have -inf at all output indices except their own;
        mask positions will have 0.0 at all non-special outputs.

        Args:
            logits_SPT: Raw model logits, shape (S, P, T).
            seq_SP: Input token ids, shape (S, P).

        Returns:
            Constrained logits as float32, shape (S, P, To).
        """
        output_mask_SPTo = self.valid_output_mask_TiTo[seq_SP]
        return logits_SPT.float() + output_mask_SPTo


class MPNNTokenizer:
    """Tokenizer using ProteinMPNN's amino acid vocabulary.

    Maps single-letter amino acid sequences to/from PMPNN token indices.
    Default vocabulary: 20 standard amino acids + UNK (X), indexed 0-20.
    Optionally appends a ``<mask>`` token as an extra ID for guidance setups
    that need explicit mask semantics at the tokenizer level.

    Follows HuggingFace tokenizer conventions:
        - encode(sequence) -> list[int]
        - decode(token_ids) -> str
        - __call__(sequences) -> dict with 'input_ids' tensor
        - vocab_size property
    """

    def __init__(
        self,
        include_mask_token: bool = False,
        mask_token: str = "<mask>",
    ):
        from atomworks.constants import DICT_THREE_TO_ONE, UNKNOWN_AA
        from mpnn.transforms.feature_aggregation.token_encodings import (
            MPNN_TOKEN_ENCODING,
        )

        three_to_idx = MPNN_TOKEN_ENCODING.token_to_idx

        # Build one-letter <-> index mappings
        self._one_to_idx: dict[str, int] = {}
        self._idx_to_one: dict[int, str] = {}
        for three_letter, idx in three_to_idx.items():
            one_letter = DICT_THREE_TO_ONE.get(
                str(three_letter), DICT_THREE_TO_ONE[UNKNOWN_AA]
            )
            self._one_to_idx[one_letter] = int(idx)
            self._idx_to_one[int(idx)] = one_letter

        self.unk_token = "X"
        self.unk_token_id = self._one_to_idx[self.unk_token]
        self._vocab_size = len(three_to_idx)

        self.mask_token = None
        self.mask_token_id = None
        if include_mask_token:
            if mask_token in self._one_to_idx:
                raise ValueError(
                    f"mask_token {mask_token!r} already exists in PMPNN vocabulary"
                )
            self.mask_token = mask_token
            self.mask_token_id = self._vocab_size
            self._one_to_idx[mask_token] = self.mask_token_id
            self._idx_to_one[self.mask_token_id] = mask_token
            self._vocab_size += 1

        # HF-compatible attributes for interop with guidance/tokenizer utilities.
        self.cls_token_id = None
        self.eos_token_id = None
        self.pad_token_id = None
        self.added_tokens_decoder: dict[int, str] = {}  # no special tokens

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def vocab(self) -> dict[str, int]:
        """Token-to-index mapping, compatible with HF tokenizer interface."""
        return dict(self._one_to_idx)

    def encode(self, sequence: str) -> list[int]:
        """Convert a single-letter AA sequence to token indices."""
        return [self._one_to_idx.get(aa, self.unk_token_id) for aa in sequence]

    def decode(self, token_ids: list[int] | torch.Tensor) -> str:
        """Convert token indices back to a single-letter AA sequence."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return "".join(self._idx_to_one.get(idx, self.unk_token) for idx in token_ids)

    def __call__(self, sequences: str | list[str]) -> dict[str, torch.Tensor]:
        """Tokenize one or more sequences, returning a dict with 'input_ids' tensor."""
        if isinstance(sequences, str):
            sequences = [sequences]
        input_ids = torch.tensor(
            [self.encode(seq) for seq in sequences], dtype=torch.long
        )
        return {"input_ids": input_ids}
