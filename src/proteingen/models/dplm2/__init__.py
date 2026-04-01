from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModelForMaskedLM, EsmTokenizer

from proteingen.generative_modeling import (
    TransitionModelWithEmbedding,
    MaskedModelLogitFormatter,
)


class DPLM2Tokenizer:
    """Tokenizer for DPLM2's extended vocabulary (AA + structure tokens).

    Wraps HuggingFace's EsmTokenizer with DPLM2-specific special token assignments.
    The DPLM2 vocabulary has 3 regions:
        - Tokens 0-32: amino acid tokens + AA special tokens
        - Tokens 33-8228: structure tokens + struct special tokens
        - Token IDs >= vocab_size: generic HF special tokens (excluded)

    Key special tokens:
        - 0: <cls_aa>  (BOS for amino acids)
        - 1: <pad>
        - 2: <eos_aa>  (EOS for amino acids)
        - 32: <mask_aa> (mask for amino acid diffusion)
    """

    # Special token IDs in the DPLM2 vocabulary
    _AA_SPECIAL = {0, 1, 2, 3, 32}  # cls_aa, pad, eos_aa, unk_aa, mask_aa
    _STRUCT_SPECIAL = {
        33,
        34,
        35,
        8228,
    }  # cls_struct, eos_struct, unk_struct, mask_struct
    _NON_STANDARD_AA = {24, 25, 26, 27, 28}  # X, B, U, Z, O
    _OTHER = {29, 30, 31}  # '.', '-', '<null_1>'

    def __init__(self, checkpoint: str, vocab_size: int):
        self._tok = EsmTokenizer.from_pretrained(checkpoint)
        # Filter vocab to model's actual vocab_size (excludes generic HF special tokens)
        full_vocab = self._tok.get_vocab()
        self._vocab = {k: v for k, v in full_vocab.items() if v < vocab_size}
        self._vocab_size = vocab_size
        self._id_to_tok = {v: k for k, v in self._vocab.items()}

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    @property
    def mask_token_id(self) -> int:
        return 32  # <mask_aa>

    @property
    def cls_token_id(self) -> int:
        return 0  # <cls_aa>

    @property
    def eos_token_id(self) -> int:
        return 2  # <eos_aa>

    @property
    def pad_token_id(self) -> int:
        return 1  # <pad>

    @property
    def added_tokens_decoder(self) -> dict[int, str]:
        all_special = (
            self._AA_SPECIAL
            | self._STRUCT_SPECIAL
            | self._NON_STANDARD_AA
            | self._OTHER
        )
        return {i: self._id_to_tok[i] for i in all_special if i in self._id_to_tok}

    @property
    def all_special_ids(self) -> list[int]:
        return sorted(self._AA_SPECIAL | self._STRUCT_SPECIAL)

    def encode(self, sequence: str, add_special_tokens: bool = True) -> list[int]:
        """Encode an amino acid sequence to token IDs."""
        token_ids = [self._vocab.get(c, 3) for c in sequence]  # 3 = unk_aa
        if add_special_tokens:
            token_ids = [self.cls_token_id] + token_ids + [self.eos_token_id]
        return token_ids

    def decode(self, token_ids: list[int] | torch.Tensor) -> str:
        """Decode token IDs back to an amino acid sequence (skipping special tokens)."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        skip = self._AA_SPECIAL | self._STRUCT_SPECIAL
        return "".join(self._id_to_tok.get(i, "X") for i in token_ids if i not in skip)

    def __call__(
        self,
        sequences: str | list[str],
        padding: bool = False,
        return_tensors: str | None = None,
    ) -> dict[str, list | torch.Tensor]:
        if isinstance(sequences, str):
            sequences = [sequences]
        encoded = [self.encode(seq) for seq in sequences]
        if padding:
            max_len = max(len(e) for e in encoded)
            encoded = [e + [self.pad_token_id] * (max_len - len(e)) for e in encoded]
        result = {"input_ids": encoded}
        if return_tensors == "pt":
            result["input_ids"] = torch.tensor(result["input_ids"], dtype=torch.long)
        return result


class DPLM2(TransitionModelWithEmbedding):
    """DPLM-2 discrete diffusion protein language model.

    Wraps ByteDance's DPLM-2 (multimodal diffusion protein LM) as a
    TransitionModelWithEmbedding for use with proteingen's sampling,
    guidance, and probe infrastructure.

    Currently supports sequence-only mode: input is [<cls_aa>, AA..., <eos_aa>].
    Structure conditioning (joint sequence + structure token input) is not yet
    implemented.

    Available checkpoints:
        - ``"airkingbd/dplm2_150m"`` — 150M params, 640d, 30 layers
        - ``"airkingbd/dplm2_650m"`` — 650M params, 1280d, 33 layers (default)
        - ``"airkingbd/dplm2_3b"``   — 3B params, 2560d, 36 layers

    Tensor index legend:
        S: batch index
        P: position index
        T: token/vocab dimension (OUTPUT_DIM = vocab_size = 8229)
        D: embedding dimension (EMB_DIM = hidden_size)

    Example::

        model = DPLM2("airkingbd/dplm2_650m")
        log_probs = model.get_log_probs_from_string(["ACDEF"])
    """

    # Token dropout rate hardcoded in ESM training — needed to match forward path
    _MASK_RATIO_TRAIN = 0.15 * 0.8  # = 0.12

    def __init__(self, checkpoint: str = "airkingbd/dplm2_650m"):
        self._checkpoint = checkpoint

        config = AutoConfig.from_pretrained(checkpoint)
        # DPLM2 was trained with untied embedding/decoder weights.
        # The HF config incorrectly says tie_word_embeddings=True.
        config.tie_word_embeddings = False

        tokenizer = DPLM2Tokenizer(checkpoint, vocab_size=config.vocab_size)
        model = AutoModelForMaskedLM.from_pretrained(checkpoint, config=config).eval()

        self.OUTPUT_DIM = config.vocab_size
        self.EMB_DIM = config.hidden_size

        logit_formatter = MaskedModelLogitFormatter(tokenizer, self.OUTPUT_DIM)

        super().__init__(
            model=model, tokenizer=tokenizer, logit_formatter=logit_formatter
        )

    def _save_args(self) -> dict:
        return {"checkpoint": self._checkpoint}

    def differentiable_embedding(
        self, ohe_seq_SPT: torch.FloatTensor
    ) -> torch.FloatTensor:
        """OHE (or soft distribution) → deep embeddings through the transformer.

        Replicates the forward path through EsmEmbeddings + EsmEncoder:
        1. Soft word embedding lookup via matmul
        2. Token dropout: zero mask positions, then rescale (ESM's mask-ratio
           compensation, runs in both train and eval mode)
        3. Attention mask application
        4. Transformer encoder with rotary attention

        For soft distributions (TAG guidance), mask tokens should not appear
        in the input — the zeroing step uses argmax to identify mask positions,
        which is non-differentiable at those positions.
        """
        # Soft word embedding lookup
        x = ohe_seq_SPT @ self.model.esm.embeddings.word_embeddings.weight  # (S, P, D)

        # Token dropout: zero out mask positions, then compensate by rescaling.
        # ESM always applies this, even in eval mode.
        # For non-masked inputs: scale = (1 - 0.12) / (1 - 0) = 0.88
        pseudo_ids = ohe_seq_SPT.argmax(-1)
        attention_mask = pseudo_ids.ne(self.tokenizer.pad_token_id)
        is_mask = pseudo_ids == self.tokenizer.mask_token_id
        x = x.masked_fill(is_mask.unsqueeze(-1), 0.0)
        src_lengths = attention_mask.sum(-1)
        mask_ratio_observed = is_mask.sum(-1).float() / src_lengths
        x = (
            x * (1 - self._MASK_RATIO_TRAIN) / (1 - mask_ratio_observed)[:, None, None]
        ).to(x.dtype)

        # Zero out padding positions (same as EsmEmbeddings.forward)
        x = (x * attention_mask.unsqueeze(-1)).to(x.dtype)

        # Build extended attention mask for the encoder
        extended_mask = self.model.esm.get_extended_attention_mask(
            attention_mask, pseudo_ids.shape
        )
        head_mask = self.model.esm.get_head_mask(
            None, self.model.config.num_hidden_layers
        )

        # Run through transformer encoder
        encoder_out = self.model.esm.encoder(
            x,
            attention_mask=extended_mask,
            head_mask=head_mask,
        )
        return encoder_out.last_hidden_state

    def embedding_to_outputs(
        self, embedding_SPD: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Deep embeddings → logits via the LM head."""
        return self.model.lm_head(embedding_SPD)

    def format_raw_to_logits(
        self, raw_output, seq_SP: torch.LongTensor, **kwargs
    ) -> torch.FloatTensor:
        """Extract logits from MaskedLMOutput and apply logit formatting."""
        logits_SPT = raw_output.logits.float()
        return self.logit_formatter(logits_SPT, seq_SP)
