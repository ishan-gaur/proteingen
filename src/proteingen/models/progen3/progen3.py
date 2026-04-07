"""ProGen3 autoregressive protein language model wrapper.

Wraps Profluent's ProGen3 family of sparse mixture-of-experts causal language
models for protein generation and scoring. Unlike the masked models (ESMC, DPLM2),
ProGen3 generates proteins autoregressively (left-to-right).

Requires the ``progen3`` package: ``pip install git+https://github.com/Profluent-AI/progen3.git``

Tensor Index Legend:
    S: batch (sequence) index
    P: position index within a sequence (includes BOS + direction + AA + direction + EOS)
    T: token/vocab dimension (OUTPUT_DIM = 134)
    D: embedding dimension (EMB_DIM, set dynamically from model weights)
"""

from __future__ import annotations

import re
import sys
import types
from typing import Any, TypedDict

import torch
from torch import nn

from proteingen.generative_modeling import (
    GenerativeModelWithEmbedding,
    LogitFormatter,
)


def _ensure_flash_attn_mock():
    """Install a pure-PyTorch fallback for flash_attn's Triton RMSNorm.

    progen3.modeling imports ``flash_attn.ops.triton.layer_norm.rms_norm_fn``
    at module level. The ``flash_attn`` pip package requires GLIBC ≥ 2.32
    because its CUDA extension is compiled against it — but progen3 only uses
    the Triton-based RMSNorm (no CUDA kernels). This function installs a
    pure-PyTorch shim so progen3 can load on systems with older GLIBC.

    The shim is only installed if ``flash_attn`` is not already importable.
    """
    try:
        import flash_attn  # noqa: F401

        return  # real package works, nothing to do
    except (ImportError, OSError):
        pass  # GLIBC or missing package — install shim

    def _rms_norm_fn(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        residual: torch.Tensor | None = None,
        eps: float = 1e-6,
        dropout_p: float = 0.0,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ) -> torch.Tensor:
        """Pure-PyTorch RMSNorm matching the flash_attn Triton kernel's API."""
        input_dtype = x.dtype
        if residual is not None:
            x = x + residual
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        x = (x * weight).to(input_dtype)
        if bias is not None:
            x = x + bias
        return x

    # Build the module hierarchy: flash_attn.ops.triton.layer_norm
    fa = types.ModuleType("flash_attn")
    fa.__version__ = "0.0.0"  # type: ignore[attr-defined]

    # progen3 needs: flash_attn.ops.triton.layer_norm.rms_norm_fn
    fa_ops = types.ModuleType("flash_attn.ops")
    fa_ops_triton = types.ModuleType("flash_attn.ops.triton")
    fa_ops_triton_ln = types.ModuleType("flash_attn.ops.triton.layer_norm")
    fa_ops_triton_ln.rms_norm_fn = _rms_norm_fn  # type: ignore[attr-defined]
    fa.ops = fa_ops  # type: ignore[attr-defined]
    fa_ops.triton = fa_ops_triton  # type: ignore[attr-defined]
    fa_ops_triton.layer_norm = fa_ops_triton_ln  # type: ignore[attr-defined]

    # transformers checks is_flash_attn_2_available() at import time and,
    # if True, imports from flash_attn.bert_padding and flash_attn itself.
    # The real flash_attn 2.8.3 is pip-installed (metadata exists) but its
    # CUDA extension can't load (GLIBC too old). Provide dummy stubs so
    # transformers' module-level imports succeed. These are never called —
    # progen3 uses PyTorch SDPA, not flash_attn_func.
    def _not_implemented(*args, **kwargs):
        raise NotImplementedError(
            "flash_attn CUDA kernels unavailable; using SDPA backend"
        )

    fa_bert = types.ModuleType("flash_attn.bert_padding")
    fa_bert.index_first_axis = _not_implemented  # type: ignore[attr-defined]
    fa_bert.pad_input = _not_implemented  # type: ignore[attr-defined]
    fa_bert.unpad_input = _not_implemented  # type: ignore[attr-defined]
    fa.bert_padding = fa_bert  # type: ignore[attr-defined]

    # flash_attn_func / flash_attn_varlen_func stubs
    fa.flash_attn_func = _not_implemented  # type: ignore[attr-defined]
    fa.flash_attn_varlen_func = _not_implemented  # type: ignore[attr-defined]

    # Set __spec__ on all mocks so importlib.util.find_spec doesn't crash
    from importlib.machinery import ModuleSpec

    all_mocks = {
        "flash_attn": fa,
        "flash_attn.ops": fa_ops,
        "flash_attn.ops.triton": fa_ops_triton,
        "flash_attn.ops.triton.layer_norm": fa_ops_triton_ln,
        "flash_attn.bert_padding": fa_bert,
    }
    for name, mod in all_mocks.items():
        mod.__spec__ = ModuleSpec(name, None)
        sys.modules[name] = mod


# ── Logit Formatter ──────────────────────────────────────────────────────


class AutoregressiveLogitFormatter(nn.Module, LogitFormatter):
    """Logit formatter for autoregressive models used with ``sample()``.

    In the proteingen framework, ``logits[i]`` is the prediction for position i.
    Autoregressive models naturally produce ``logits[i]`` = prediction for position
    i+1. The caller (``format_raw_to_logits``) shifts the logits before passing
    them here so they are already position-aligned.

    This formatter then applies:

    - **Non-mask positions**: one-hot (predict themselves — already decoded or
      fixed framing tokens).
    - **First mask position** (leftmost per sequence): pass through the model's
      logits, restricted to standard amino acids.
    - **Remaining mask positions**: all ``-inf`` (not yet reachable by the
      autoregressive model).

    This makes the model compatible with ``sample(model, x, in_order="left_to_right")``.

    Tensor Index Legend:
        Ti: input token index — rows of the one-hot mask, size = vocab_size
        To: output token index — columns, size = output_dim
    """

    STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")

    def __init__(self, tokenizer: _ProGen3TokenizerAdapter, output_dim: int):
        nn.Module.__init__(self)
        self.tokenizer = tokenizer
        self.output_dim = output_dim

        PASS, BLOCK = 0.0, float("-inf")

        # One-hot mask for non-fill positions: (Ti, To)
        # Each token predicts only itself
        one_hot_mask = torch.full(
            (tokenizer.vocab_size, output_dim), BLOCK, dtype=torch.float32
        )
        for idx in range(tokenizer.vocab_size):
            if idx < output_dim:
                one_hot_mask[idx, idx] = PASS
        self.register_buffer("one_hot_mask_TiTo", one_hot_mask)

        # AA-only additive mask for fill positions: (To,)
        # Allows only standard amino acid outputs
        aa_mask = torch.full((output_dim,), BLOCK, dtype=torch.float32)
        for tok, idx in tokenizer.vocab.items():
            if tok in self.STANDARD_AAS:
                aa_mask[idx] = PASS
        self.register_buffer("aa_mask_To", aa_mask)

    def forward(
        self, logits_SPT: torch.Tensor, seq_SP: torch.LongTensor
    ) -> torch.FloatTensor:
        S, P, T = logits_SPT.shape
        mask_id = self.tokenizer.mask_token_id
        is_mask = seq_SP == mask_id  # [S, P]

        # Start with one-hot for all positions
        result = self.one_hot_mask_TiTo[seq_SP]  # [S, P, To]

        if is_mask.any():
            # Block ALL mask positions (override one-hot for mask token)
            result[is_mask] = float("-inf")

            # Pass through logits only at the first mask position per sequence
            has_mask = is_mask.any(dim=1)  # [S]
            first_mask_pos = is_mask.float().argmax(dim=1)  # [S]

            batch_idx = torch.arange(S, device=seq_SP.device)[has_mask]
            pos_idx = first_mask_pos[has_mask]

            # Model logits + AA restriction at first mask position
            result[batch_idx, pos_idx] = (
                logits_SPT[batch_idx, pos_idx] + self.aa_mask_To
            )

        return result.float()


# ── Tokenizer Adapter ────────────────────────────────────────────────────


class _ProGen3TokenizerAdapter:
    """Adapts progen3's ``tokenizers.Tokenizer`` to the HF-compatible interface
    expected by ``GenerativeModel``.

    The progen3 tokenizer uses the ``tokenizers`` library (not HF
    ``PreTrainedTokenizer``). This adapter provides the subset of the HF API
    that ``GenerativeModelWithEmbedding``, ``AutoregressiveLogitFormatter``,
    ``sample()``, and ``get_log_probs_from_string`` rely on.

    Encoding convention:
        Raw sequence ``"ACDE"`` is encoded as ``<bos> 1 A C D E 2 <eos>``
        where ``1`` = N-to-C direction token, ``2`` = C-to-N direction token.

    For ``sample()`` integration, ``<mask>`` tokens mark positions to be filled:
        ``"<mask><mask><mask>"`` → ``<bos> 1 <mask> <mask> <mask> 2 <eos>``
    """

    # Standard amino acid one-letter codes recognized by the model
    AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

    def __init__(self):
        from progen3.tokenizer import get_tokenizer

        self._tok = get_tokenizer()
        self._vocab = self._tok.get_vocab()

        # Core token IDs
        self.pad_token_id: int = self._vocab["<pad>"]
        self.bos_token_id: int = self._vocab["<bos>"]
        self.eos_token_id: int = self._vocab["<eos>"]
        self.mask_token_id: int = self._vocab["<mask>"]

        # Direction tokens
        self.n_to_c_token_id: int = self._vocab["1"]
        self.c_to_n_token_id: int = self._vocab["2"]

        # HF-compatible: mapping of special token indices to their string names
        self.added_tokens_decoder: dict[int, str] = {}
        special_pattern = re.compile(r"^<.+>$|^[12]$")
        for tok, idx in self._vocab.items():
            if special_pattern.match(tok):
                self.added_tokens_decoder[idx] = tok

        # AA token IDs (for restricting generation output)
        self.aa_token_ids: list[int] = sorted(
            idx for tok, idx in self._vocab.items() if tok in self.AMINO_ACIDS
        )

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    @property
    def all_special_ids(self) -> list[int]:
        return list(self.added_tokens_decoder.keys())

    def encode_sequence(self, sequence: str) -> list[int]:
        """Encode a raw AA sequence with BOS + direction + EOS framing.

        ``"ACDE"`` → ``[<bos>, 1, A, C, D, E, 2, <eos>]``

        Also handles ``<mask>`` tokens for ``sample()`` integration:
        ``"<mask><mask>"`` → ``[<bos>, 1, <mask>, <mask>, 2, <eos>]``
        """
        framed = f"<bos>1{sequence}2<eos>"
        return self._tok.encode(framed).ids

    def decode(self, token_ids: list[int] | torch.Tensor) -> str:
        """Decode token IDs to an amino acid string, stripping all framing."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.extract_sequence(token_ids)

    def batch_decode(self, token_ids_batch: torch.Tensor) -> list[str]:
        """Decode a batch of token ID sequences to amino acid strings."""
        return [self.decode(row) for row in token_ids_batch]

    def extract_sequence(self, token_ids: list[int]) -> str:
        """Extract the amino acid sequence from encoded token IDs, stripping framing tokens."""
        aa_set = self.AMINO_ACIDS
        decoded = self._tok.decode(token_ids, skip_special_tokens=False)
        return "".join(c for c in decoded if c in aa_set)

    def __call__(
        self,
        sequences: str | list[str],
        padding: bool = False,
        return_tensors: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Tokenize sequences with optional padding. HF-compatible interface."""
        if isinstance(sequences, str):
            sequences = [sequences]

        encoded = [self.encode_sequence(seq) for seq in sequences]

        if padding:
            max_len = max(len(e) for e in encoded)
            encoded = [e + [self.pad_token_id] * (max_len - len(e)) for e in encoded]

        input_ids = torch.tensor(encoded, dtype=torch.long)
        return {"input_ids": input_ids}


# ── Result Types ─────────────────────────────────────────────────────────


class ProGen3GenerationResult(TypedDict):
    """Result of autoregressive protein generation."""

    sequences: list[str]


# ── Model ────────────────────────────────────────────────────────────────


class ProGen3(GenerativeModelWithEmbedding):
    """ProGen3 autoregressive protein language model.

    Wraps Profluent's ProGen3 family (112M–46B params, sparse MoE) as a
    ``GenerativeModelWithEmbedding``. Unlike the masked models in proteingen,
    ProGen3 is autoregressive — it generates proteins left-to-right (N→C terminal).

    Main use cases:

    - **Unconditional generation via sample()**: use ``<mask>`` tokens with
      ``in_order="left_to_right"``
    - **Open-ended generation**: ``model.generate()`` produces variable-length proteins
    - **Scoring**: ``model.score(["MKTLLLTL..."])`` returns per-sequence log-likelihoods
    - **Embeddings / linear probes**: ``model.embed(seq)`` returns transformer hidden
      states; the representation at the last real token position is a sequence embedding

    Available checkpoints (HuggingFace hub, ``Profluent-Bio/``):

    ==================== ====== ====== ======
    Checkpoint            Params Hidden Layers
    ==================== ====== ====== ======
    progen3-112m          112M   384    10
    progen3-3b            3B     2048   40
    ==================== ====== ====== ======

    Example::

        from proteingen.models import ProGen3
        from proteingen import sample

        model = ProGen3("Profluent-Bio/progen3-112m").cuda()

        # Fixed-length sampling via sample()
        init_x = ["<mask>" * 100 for _ in range(5)]
        result = sample(model, init_x, in_order="left_to_right")

        # Open-ended generation (variable length)
        result = model.generate(n=5, max_new_tokens=256)

    Tensor Index Legend:
        S: batch (sequence) index
        P: position index within a sequence
        T: token/vocab dimension (OUTPUT_DIM = 134)
        D: embedding dimension (EMB_DIM, varies by checkpoint)
    """

    OUTPUT_DIM = 134  # progen3 vocab size

    def __init__(
        self,
        checkpoint: str = "Profluent-Bio/progen3-112m",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        _ensure_flash_attn_mock()
        from progen3.modeling import ProGen3ForCausalLM

        model = ProGen3ForCausalLM.from_pretrained(checkpoint, torch_dtype=torch_dtype)
        model.eval()

        tokenizer = _ProGen3TokenizerAdapter()
        logit_formatter = AutoregressiveLogitFormatter(tokenizer, self.OUTPUT_DIM)

        self.EMB_DIM = model.model.embed_tokens.weight.shape[1]

        super().__init__(
            model=model, tokenizer=tokenizer, logit_formatter=logit_formatter
        )

        self._checkpoint = checkpoint
        self._torch_dtype = torch_dtype

    # ── GenerativeModelWithEmbedding interface ───────────────────────────

    def differentiable_embedding(
        self, ohe_seq_SPT: torch.FloatTensor
    ) -> torch.FloatTensor:
        """OHE (or soft distribution) over tokens → deep embeddings (S, P, D).

        Performs differentiable embedding lookup via matrix multiply, adds
        default sequence ID embedding (all zeros), then runs through the
        full transformer body with causal attention.
        """
        S, P, T = ohe_seq_SPT.shape
        device = ohe_seq_SPT.device

        # Differentiable embedding lookup (cast OHE to model dtype for matmul)
        weight_dtype = self.model.model.embed_tokens.weight.dtype
        emb_SPD = (
            ohe_seq_SPT.to(weight_dtype) @ self.model.model.embed_tokens.weight
        )  # [S, P, D]

        # Add sequence ID embedding (all zeros for single-sequence mode)
        seq_ids = torch.zeros(S, P, dtype=torch.long, device=device)  # [S, P]
        emb_SPD = emb_SPD + self.model.model.embed_seq_id(seq_ids)  # [S, P, D]

        # Cast to model dtype
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        else:
            target_dtype = weight_dtype
        hidden_states = emb_SPD.to(target_dtype)  # [S, P, D]

        # Default position_ids = sequential
        position_ids = (
            torch.arange(P, device=device).unsqueeze(0).expand(S, -1)
        )  # [S, P]

        # Run through transformer layers
        for layer in self.model.model.layers:
            layer_outputs = layer(
                hidden_states,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                output_router_weights=False,
                use_cache=False,
            )
            hidden_states = layer_outputs[0]  # [S, P, D]

        hidden_states = self.model.model.norm(hidden_states)  # [S, P, D]
        return hidden_states

    def embedding_to_outputs(
        self, embedding_SPD: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Deep embeddings → logits via the language model head."""
        return self.model.lm_head(embedding_SPD).float()  # [S, P, T=134]

    def format_raw_to_logits(
        self, raw_output: torch.FloatTensor, seq_SP: torch.LongTensor, **kwargs
    ) -> torch.FloatTensor:
        """Shift AR logits to align with positions, then apply formatter.

        Raw AR logits: ``logits[i]`` = prediction for position ``i+1`` given
        tokens ``0..i``. The framework expects ``logits[i]`` = prediction for
        position ``i``. We shift left by one, then apply the logit formatter.
        """
        S, P, T = raw_output.shape

        # Shift left: aligned[i] = raw[i-1] (prediction for position i)
        aligned_SPT = torch.zeros_like(raw_output)  # [S, P, T]
        aligned_SPT[:, 1:, :] = raw_output[
            :, :-1, :
        ]  # [S, P-1, T] shifted into [S, 1:P, T]
        # Position 0 (BOS) gets zeros — formatter will make it one-hot anyway

        return self.logit_formatter(aligned_SPT, seq_SP)

    # ── Forward (overrides GenerativeModelWithEmbedding default) ─────────

    def forward(self, seq_SP: torch.LongTensor, **kwargs) -> Any:
        """Forward pass through ProGen3.

        Automatically constructs ``position_ids`` and ``sequence_ids`` from
        the token sequence if not provided in kwargs.

        Returns raw (unshifted) logits of shape (S, P, T=134).
        """
        S, P = seq_SP.shape
        device = seq_SP.device

        position_ids = kwargs.pop(
            "position_ids",
            torch.arange(P, device=device).unsqueeze(0).expand(S, -1),
        )  # [S, P]
        sequence_ids = kwargs.pop(
            "sequence_ids",
            torch.zeros(S, P, dtype=torch.long, device=device),
        )  # [S, P]

        output = self.model(
            input_ids=seq_SP,
            position_ids=position_ids,
            sequence_ids=sequence_ids,
            return_dict=True,
            use_cache=False,
        )
        return output.logits.float()  # [S, P, T=134]

    # ── Convenience methods ──────────────────────────────────────────────

    def generate(
        self,
        prompt: str = "",
        n: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> ProGen3GenerationResult:
        """Generate protein sequences autoregressively (variable length).

        For fixed-length generation, use ``sample(model, x, in_order="left_to_right")``
        instead. This method uses HuggingFace's ``model.generate()`` and lets the
        model decide when to stop.

        Args:
            prompt: Optional amino acid prefix (N-terminal). Empty string for
                unconditional generation.
            n: Number of sequences to generate.
            max_new_tokens: Maximum number of new tokens to generate (including
                the C-terminal direction token and EOS).
            temperature: Sampling temperature. Higher = more diverse.
            top_p: Nucleus sampling threshold.

        Returns:
            Dict with ``"sequences"`` key mapping to a list of generated amino acid
            strings (framing tokens stripped).
        """
        from progen3.batch_preparer import ProGen3BatchPreparer
        from transformers import GenerationConfig

        batch_preparer = ProGen3BatchPreparer()
        tok = self.tokenizer

        # Build prefix: <bos> 1 [prompt_AAs...]
        prefix_str = f"<bos>1{prompt}"
        prefix_ids = batch_preparer.tokenizer.encode(prefix_str).ids
        input_ids = (
            torch.tensor(prefix_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        )  # [1, prefix_len]
        position_ids = torch.arange(len(prefix_ids), device=self.device).unsqueeze(
            0
        )  # [1, prefix_len]
        sequence_ids = torch.zeros_like(input_ids)  # [1, prefix_len]

        eos_token_id = batch_preparer.tokenizer.token_to_id("<eos>")

        gen_config = GenerationConfig(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            pad_token_id=tok.pad_token_id,
            num_return_sequences=n,
        )

        input_ids_expanded = input_ids.expand(n, -1)  # [n, prefix_len]
        position_ids_expanded = position_ids.expand(n, -1)  # [n, prefix_len]
        sequence_ids_expanded = sequence_ids.expand(n, -1)  # [n, prefix_len]

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids_expanded,
                position_ids=position_ids_expanded,
                sequence_ids=sequence_ids_expanded,
                generation_config=gen_config,
            )  # [n, generated_len]

        sequences = []
        for i in range(n):
            seq = tok.extract_sequence(outputs[i].tolist())
            sequences.append(seq)

        return ProGen3GenerationResult(sequences=sequences)

    def score(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        """Score protein sequences by autoregressive log-likelihood.

        Computes the mean per-residue log probability under the model's
        left-to-right factorization. Sequences are scored in both N→C and
        C→N directions and averaged (bidirectional scoring).

        Args:
            sequences: List of amino acid strings to score.

        Returns:
            Dict with:
            - ``"log_likelihood"``: (N,) tensor of mean per-residue log-likelihoods
            - ``"perplexity"``: (N,) tensor of perplexities
        """
        from progen3.batch_preparer import ProGen3BatchPreparer

        batch_preparer = ProGen3BatchPreparer()

        def _score_direction(seqs: list[str], reverse: bool) -> torch.Tensor:
            kwargs = batch_preparer.get_batch_kwargs(
                seqs, device=self.device, reverse=reverse
            )
            input_ids = kwargs["input_ids"]  # [S, P]
            labels = kwargs["labels"]  # [S, P]
            position_ids = kwargs["position_ids"]  # [S, P]
            sequence_ids = kwargs["sequence_ids"]  # [S, P]

            with torch.no_grad():
                output = self.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    sequence_ids=sequence_ids,
                    return_dict=True,
                    use_cache=False,
                )
            logits = output.logits.float()  # [S, P, T]

            shift_logits = logits[:, :-1, :].contiguous()  # [S, P-1, T]
            shift_labels = labels[:, 1:].contiguous()  # [S, P-1]

            nll = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view(shift_labels.shape)  # [S, P-1]

            mask = shift_labels != batch_preparer.pad_token_id  # [S, P-1]
            nll = (nll * mask).sum(dim=1) / mask.sum(dim=1)  # [S]
            return nll

        nll_fwd = _score_direction(sequences, reverse=False)  # [S]
        nll_rev = _score_direction(sequences, reverse=True)  # [S]
        nll_avg = (nll_fwd + nll_rev) / 2  # [S]

        return {
            "log_likelihood": -nll_avg,
            "perplexity": torch.exp(nll_avg),
        }

    # ── Unsupported / model-specific ─────────────────────────────────────

    def _save_args(self) -> dict:
        return {"checkpoint": self._checkpoint}

    def apply_lora(self, **kwargs) -> None:
        raise NotImplementedError(
            "LoRA is not supported for ProGen3 (sparse MoE architecture)"
        )
