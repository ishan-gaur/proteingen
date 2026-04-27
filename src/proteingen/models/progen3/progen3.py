"""ProGen3 autoregressive protein language model wrapper.

Wraps Profluent's ProGen3 family of sparse mixture-of-experts causal language
models for protein generation and scoring. Unlike the masked models (ESMC, DPLM2),
ProGen3 generates proteins autoregressively (left-to-right).

Requires the ``progen3`` package: ``pip install git+https://github.com/Profluent-AI/progen3.git``

Tensor Index Legend:
    S: batch (sequence) index
    P: position index within a sequence (includes BOS + direction + AA + direction + EOS)
    T: token/vocab dimension (OUTPUT_DIM = 134)
"""

from __future__ import annotations

import re
import sys
import types
from typing import TypedDict

import torch

from proteingen.generative_modeling import GenerativeModel, PassThroughLogitFormatter


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


class ProGen3GenerationResult(TypedDict):
    """Result of autoregressive protein generation."""

    sequences: list[str]


class _ProGen3TokenizerAdapter:
    """Adapts progen3's ``tokenizers.Tokenizer`` to the HF-compatible interface
    expected by ``GenerativeModel``.

    The progen3 tokenizer uses the ``tokenizers`` library (not HF
    ``PreTrainedTokenizer``). This adapter provides the subset of the HF API
    that ``GenerativeModel``, ``PassThroughLogitFormatter``, and
    ``get_log_probs_from_string`` rely on.

    Encoding convention:
        Raw sequence ``"ACDE"`` is encoded as ``<bos> 1 A C D E 2 <eos>``
        where ``1`` = N-to-C direction token, ``2`` = C-to-N direction token.
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
        self.mask_token_id: int | None = self._vocab.get("<mask>")

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
        """
        framed = f"<bos>1{sequence}2<eos>"
        return self._tok.encode(framed).ids

    def decode_ids(self, token_ids: list[int]) -> str:
        """Decode token IDs back to a string."""
        return self._tok.decode(token_ids, skip_special_tokens=False)

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


class ProGen3(GenerativeModel):
    """ProGen3 autoregressive protein language model.

    Wraps Profluent's ProGen3 family (112M–46B params, sparse MoE) as a
    ``GenerativeModel``. Unlike the masked models in proteingen, ProGen3 is
    autoregressive — it generates proteins left-to-right (N→C terminal).

    Main use cases:

    - **Unconditional generation**: ``model.generate()`` produces novel proteins
    - **Prompted generation**: ``model.generate(prompt="MKTL")`` completes a prefix
    - **Scoring**: ``model.score(["MKTLLLTL..."])`` returns per-sequence log-likelihoods

    The ``forward()`` / ``get_log_probs()`` interface is also available for
    compatibility with the proteingen framework, returning next-token log
    probabilities at each position.

    Available checkpoints (HuggingFace hub, ``Profluent-Bio/``):

    ============= ====== ====== ======
    Checkpoint     Params Hidden Layers
    ============= ====== ====== ======
    progen3-112m  112M   768    12
    progen3-3b    3B     2048   40
    ============= ====== ====== ======

    Example::

        from proteingen.models import ProGen3

        model = ProGen3("Profluent-Bio/progen3-112m").cuda()
        result = model.generate(n=5, max_new_tokens=256)
        print(result["sequences"])

    Limitations vs masked models:
        - No mask-based sampling (``sample()`` / ``sample_ctmc_linear_interpolation()`` don't apply)
        - No TAG guidance (no differentiable embedding path)
        - No LoRA support (MoE architecture + megablocks)

    Tensor Index Legend:
        S: batch (sequence) index
        P: position index within a sequence
        T: token/vocab dimension (OUTPUT_DIM = 134)
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
        formatter = PassThroughLogitFormatter()
        super().__init__(model=model, tokenizer=tokenizer, logit_formatter=formatter)

        self._checkpoint = checkpoint
        self._torch_dtype = torch_dtype

    def forward(self, seq_SP: torch.LongTensor, **kwargs) -> torch.FloatTensor:
        """Forward pass through ProGen3.

        Automatically constructs ``position_ids`` and ``sequence_ids`` from
        the token sequence if not provided in kwargs.

        Args:
            seq_SP: Token IDs of shape (S, P), including BOS/direction/EOS framing.
            **kwargs: Optional ``position_ids``, ``sequence_ids`` overrides.

        Returns:
            Logits tensor of shape (S, P, T=134).
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

    def generate(
        self,
        prompt: str = "",
        n: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> ProGen3GenerationResult:
        """Generate protein sequences autoregressively.

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

        # EOS = <eos> token signals the model to stop (the "2" direction
        # token precedes it in a valid completion)
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

        # Expand inputs for n sequences
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

        # Extract amino acid sequences from generated tokens
        sequences = []
        for i in range(n):
            generated_ids = outputs[i].tolist()
            seq = tok.extract_sequence(generated_ids)
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
            """Compute per-sequence NLL for one direction."""
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

            # Shift: logits[i] predicts token[i+1]
            shift_logits = logits[:, :-1, :].contiguous()  # [S, P-1, T]
            shift_labels = labels[:, 1:].contiguous()  # [S, P-1]

            nll = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view(shift_labels.shape)  # [S, P-1]

            # Mask out padding
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

    # ── Unsupported operations ───────────────────────────────────────────

    def _save_args(self) -> dict:
        return {"checkpoint": self._checkpoint}

    def apply_lora(self, **kwargs) -> None:
        raise NotImplementedError(
            "LoRA is not supported for ProGen3 (sparse MoE architecture)"
        )
