from pathlib import Path

import torch
from torch import nn
from typing import Any, Optional
from abc import ABC, abstractmethod
from contextlib import contextmanager
from .generative_modeling import GenerativeModelWithEmbedding
from .probability_model import ProbabilityModel
from transformers import PreTrainedTokenizerBase


class PredictiveModel(ProbabilityModel, ABC):
    """Base class for predictive models used in guidance.

    Predictive models answer: "what is log p(target_event | sequence)?"

    The target event is set via ``set_target_()`` or the ``with_target()`` context
    manager. ``forward()`` returns raw predictions (class logits, regression
    values, etc.). ``format_raw_to_logits()`` converts those to binary logits
    ``(B, 2)``: ``[false_logit, true_logit]``. The inherited
    ``ProbabilityModel.get_log_probs`` applies temperature-scaled log_softmax,
    and this class's override takes ``[:, 1]`` to return the scalar
    log p(target=True | x).

    Pipeline::

        get_log_probs(seq_SP) — creates OHE, stashes self._ohe
            ↓
        forward(ohe_seq) → raw output (class logits, regression value, ...)
            ↓
        format_raw_to_logits(raw) → (B, 2) binary logits [false, true]
            ↓
        ProbabilityModel.get_log_probs: log_softmax(logits / temp) → (B, 2)
            ↓
        PredictiveModel.get_log_probs: [:, 1] → (B,) log p(target | x)

    For TAG guidance, use ``grad_log_prob(seq_SP)`` which runs the pipeline,
    backprops, and returns gradients w.r.t. the model's OHE feature space.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__()
        self.tokenizer = tokenizer
        self.target = None
        self._ohe = None

    def set_target_(self, target):
        """Set the target event in-place."""
        self.target = target

    def set_target(self, target):
        """Set the target event, returning self for chaining."""
        self.set_target_(target)
        return self

    @contextmanager
    def with_target(self, target_spec):
        """Context manager: temporarily set target, revert on exit."""
        old = self.target
        self.set_target_(target_spec)
        try:
            yield self
        finally:
            self.target = old

    def forward(self, ohe_seq_SPT: torch.FloatTensor, **kwargs) -> Any:
        """Forward pass to produce raw predictions. To be implemented by subclasses. Must take OHE as input.
        OHE dimension is defined by ``tokens_to_ohe`` / ``token_ohe_basis``."""
        ...

    # TODO[pi] add the decorator which gives this instance-variable semantics
    def token_ohe_basis(self) -> torch.FloatTensor:
        """Return token-id → OHE feature matrix (T, K).

        Default is identity: each token id maps to its one-hot basis vector.
        Override when token ids should map to a reduced OHE space (e.g. an
        explicit mask token represented as an all-zero feature vector).
        """
        vocab_size = self.tokenizer.vocab_size
        return torch.eye(vocab_size, dtype=torch.float32)

    def tokens_to_ohe(self, seq_SP: torch.LongTensor) -> torch.FloatTensor:
        """Map token IDs to model OHE features using ``token_ohe_basis``."""
        basis_TK = self.token_ohe_basis().to(seq_SP.device)
        return basis_TK[seq_SP.long()]

    # TODO[pi] The relationship between target and format_raw_to_logits is not
    # obvious from the interface. A user subclassing OneHotMLP has to just *know*
    # that self.target exists, what type it should be, and that they need to use
    # it inside format_raw_to_logits. The binary logit functions help but still
    # require the user to pass self.target manually. Consider ways to make this
    # more discoverable — e.g. requiring subclasses to declare expected target
    # type, or having format_raw_to_logits receive target as an explicit argument
    # instead of reading it off self.
    @abstractmethod
    def format_raw_to_logits(
        self, raw_output: Any, seq_SPT: torch.FloatTensor, **kwargs
    ) -> torch.FloatTensor:
        """Convert raw predictions → binary logits (B, 2): [false_logit, true_logit].

        Uses ``self.target`` to determine what event is being evaluated.
        The parent's ``get_log_probs`` applies ``log_softmax(logits / temp)``
        on top.
        """
        ...

    def get_log_probs(self, seq_SP: torch.LongTensor) -> torch.FloatTensor:
        """Return log p(target=True | x), scalar per sequence.

        Creates model OHE features from token IDs, stashes them as ``self._ohe``
        (for gradient access via ``grad_log_prob``), then runs the parent
        pipeline (forward → format_raw_to_logits → log_softmax) to get
        (B, 2) and returns [:, 1].
        """
        assert self.target is not None, (
            "Target not set. Call set_target_() or use with_target() context manager."
        )
        ohe_seq_SPT = self.tokens_to_ohe(seq_SP).float()
        ohe_seq_SPT.requires_grad_(True)
        self._ohe = ohe_seq_SPT
        log_probs_B2 = super().get_log_probs(ohe_seq_SPT)  # (B, 2)
        assert log_probs_B2.shape[1] == 2, (
            f"Expected binary logits (B, 2) from format_raw_to_logits, got shape {log_probs_B2.shape}"
        )
        return log_probs_B2[:, 1]  # (B,)

    def predict(self, seq_SP: torch.LongTensor) -> Any:
        """Get raw model predictions from token IDs.

        Creates OHE and calls forward — returns whatever forward returns
        (scalar predictions, class logits, etc.) without binary logit
        conversion. Use for training (e.g. MSE loss) and scoring.
        """
        ohe = self.tokens_to_ohe(seq_SP).float()
        return self.forward(ohe)

    def grad_log_prob(self, seq_SP: torch.LongTensor) -> torch.FloatTensor:
        """Return gradient of log p(target | x) w.r.t. model OHE features.

        Runs ``get_log_probs`` (which creates and stashes the OHE),
        backprops, and returns ``self._ohe.grad`` of shape (B, P, K).
        This is the gradient signal TAG adds to generative model logits.
        """
        with torch.enable_grad():
            log_p = self.get_log_probs(seq_SP)
            log_p.sum().backward()
            return self._ohe.grad


# ── Binary logit conversion functions ────────────────────────────────────────
#
# Use these in your format_raw_to_logits implementation to convert raw model
# output into the (B, 2) binary logits [false_logit, true_logit] that the
# PredictiveModel pipeline expects.


def categorical_binary_logits(
    logits_BC: torch.FloatTensor, target_class: int
) -> torch.FloatTensor:
    """Multi-class logits (B, C) → binary logits (B, 2) for a target class.

    true_logit = logits[:, target_class]
    false_logit = logsumexp(logits for non-target classes)
    """
    target_logit_B = logits_BC[:, target_class]
    C = logits_BC.shape[-1]
    mask = torch.ones(C, dtype=torch.bool, device=logits_BC.device)
    mask[target_class] = False
    false_logit_B = torch.logsumexp(logits_BC[:, mask], dim=-1)
    return torch.stack([false_logit_B, target_logit_B], dim=-1)  # (B, 2)


def binary_logits(logit_B: torch.FloatTensor, target: bool = True) -> torch.FloatTensor:
    """Single logit → binary logits (B, 2) via sigmoid(x) = softmax([0, x])[1].

    If target is False, swaps the logits so [:, 1] gives P(negative).
    """
    logit_B = logit_B.reshape(-1)
    zero_B = torch.zeros_like(logit_B)
    if target:
        return torch.stack([zero_B, logit_B], dim=-1)  # (B, 2)
    else:
        return torch.stack([logit_B, zero_B], dim=-1)  # (B, 2)


def point_estimate_binary_logits(
    pred_B: torch.FloatTensor, threshold: float, k: float = 100.0
) -> torch.FloatTensor:
    """Scalar prediction → binary logits (B, 2) via steep sigmoid.

    sigmoid(k * (pred - threshold)) approximates a step function.
    Gradients unstable through the threshold though — use DEG, not TAG.
    """
    pred_B = pred_B.reshape(-1)
    logit_B = k * (pred_B - threshold)
    zero_B = torch.zeros_like(logit_B)
    return torch.stack([zero_B, logit_B], dim=-1)  # (B, 2)


def gaussian_binary_logits(
    mu_B: torch.FloatTensor, log_var_B: torch.FloatTensor, threshold: float
) -> torch.FloatTensor:
    """Gaussian (mean, log_var) → binary logits (B, 2) via CDF log-odds.

    P(Y > threshold) = Phi((mu - threshold) / sigma).
    If you want P(Y < threshold) for your application, just swap the order of the logits in your
    format_raw_to_logits implementation.
    Differentiable through both mean and variance — works with TAG.
    """
    sigma_B = (log_var_B / 2).exp()
    z_B = (threshold - mu_B) / sigma_B
    log_p_above = torch.special.log_ndtr(-z_B)
    log_p_below = torch.special.log_ndtr(z_B)
    return torch.stack([log_p_below, log_p_above], dim=-1)  # (B, 2)


# ── Template predictive models ──────────────────────────────────────────────
#
# Subclass these and implement format_raw_to_logits using the functions above.
# See protstar/models/ and examples/ for concrete usage.


class LinearProbe(PredictiveModel, ABC):
    """
    Linear probe on top of pre-computed embeddings.

    Tensor Dimension Labels:
        I: batch index
        D: embedding dimension
        O: output dimension
    """

    def __init__(
        self,
        embed_model: GenerativeModelWithEmbedding,
        output_dim: int,
        pooling_fn: Optional[callable] = None,
        freeze_embed_model: bool = True,
    ):
        super().__init__(tokenizer=embed_model.tokenizer)
        self.embed_model = embed_model
        self.embedding_dim = embed_model.EMB_DIM
        self.output_dim = output_dim
        self.w = nn.Linear(self.embedding_dim, self.output_dim)

        def _mean_pool_non_padding(emb_SPD, seq_SP):
            mask = (seq_SP != self.tokenizer.pad_token_id).unsqueeze(-1).float()
            return (emb_SPD * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        self.pooling_fn = pooling_fn or _mean_pool_non_padding

        if freeze_embed_model:
            for p in self.embed_model.parameters():
                p.requires_grad = False

    def forward(self, ohe_seq_SPT: torch.FloatTensor) -> torch.FloatTensor:
        """Full forward: differentiable embed → pool → linear."""
        emb_SPD = self.embed_model.differentiable_embedding(ohe_seq_SPT)
        seq_SP = ohe_seq_SPT.argmax(dim=-1)
        pooled_SD = self.pooling_fn(emb_SPD, seq_SP)
        return self.w(pooled_SD)

    @torch.no_grad()
    def precompute_embeddings(
        self,
        sequences: list[str],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Pre-compute pooled embeddings for training. Shape (N, EMB_DIM)."""
        self.embed_model.to(device)
        tokenizer = self.embed_model.tokenizer
        all_embeddings = []
        n = len(sequences)
        for start in range(0, n, batch_size):
            batch_seqs = sequences[start : start + batch_size]
            token_ids = tokenizer(batch_seqs, padding=True, return_tensors="pt")[
                "input_ids"
            ].to(device)
            emb_SPD = self.embed_model.embed(token_ids)
            pooled = self.pooling_fn(emb_SPD, token_ids)
            all_embeddings.append(pooled.cpu())
            if (start // batch_size) % 50 == 0:
                print(f"  Embedded {min(start + batch_size, n):>6d} / {n}")
        return torch.cat(all_embeddings, dim=0)

    # ── Checkpointing ────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save probe to a directory: config.json, head.pt, and embed_model/.

        Subclasses must implement ``_save_args()`` returning constructor kwargs.
        The embed_model is saved to a subdirectory via its own ``save()`` method.
        """
        path = Path(path)
        super().save(path)
        torch.save(self.w.state_dict(), path / "head.pt")
        self.embed_model.save(path / "embed_model")

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "LinearProbe":
        """Load probe from a directory.

        Reconstructs the object from config.json (calls ``cls(**args)``),
        loads the LoRA adapter onto the embed_model if present, and loads
        the head weights.
        """
        path = Path(path)
        obj = super().from_checkpoint(path)
        embed_path = path / "embed_model"
        if (embed_path / "lora_adapter").exists():
            obj.embed_model.load_lora(embed_path / "lora_adapter")
        obj.w.load_state_dict(torch.load(path / "head.pt", weights_only=True))
        return obj


class OneHotMLP(PredictiveModel, ABC):
    """MLP operating on one-hot encoded sequences.

    Receives OHE input from PredictiveModel.get_log_probs, flattens across
    all positions, and passes through an MLP. Subclasses implement
    format_raw_to_logits to convert the MLP output to binary logits.

    Tensor Dimension Labels:
        S: batch (sample) index
        P: position in sequence
        T: token dimension (one-hot / vocab size)
        O: output dimension
    """

    def __init__(
        self,
        tokenizer,
        sequence_length: int,
        model_dim: int,
        n_layers: int,
        output_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__(tokenizer)
        self.vocab_size = tokenizer.vocab_size
        self.sequence_length = sequence_length
        self.model_dim = model_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.dropout = dropout

        layers: list[nn.Module] = [
            nn.Linear(self.sequence_length * self.vocab_size, self.model_dim)
        ]
        for _ in range(n_layers - 1):
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(self.model_dim, self.model_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(self.model_dim, self.output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, ohe_seq_SPT: torch.FloatTensor) -> torch.FloatTensor:
        x_SPxT = ohe_seq_SPT.reshape(ohe_seq_SPT.size(0), -1)
        return self.layers(x_SPxT)


class PairwiseLinearModel(PredictiveModel, ABC):
    """Linear model that uses single and pairwise mutation features
    encoded as one-hot vectors.

    Receives OHE input from PredictiveModel.get_log_probs, flattens across
    all positions, and passes through an MLP. Subclasses implement
    format_raw_to_logits to convert the MLP output to binary logits.

    Tensor Dimension Labels:
        S: batch (sample) index
        P: position in sequence
        T: token dimension (one-hot / vocab size)
        O: output dimension
    """

    def __init__(
        self,
        tokenizer,
        sequence_length: int,
        output_dim: int,
    ):
        super().__init__(tokenizer)
        self.vocab_size = tokenizer.vocab_size
        self.sequence_length = sequence_length
        self.n_ohe_features = self.sequence_length * self.vocab_size + 1
        self.n_pairwise_features = torch.triu_indices(
            self.n_ohe_features, self.n_ohe_features
        ).shape[1]
        self.linear_layer = nn.Linear(self.n_pairwise_features, output_dim)

    def forward(self, ohe_seq_SPT: torch.FloatTensor) -> torch.FloatTensor:
        x_SPxT = ohe_seq_SPT.reshape(ohe_seq_SPT.size(0), -1)
        ones_S1 = torch.ones_like(x_SPxT[:, :1])
        x_Sf = torch.cat([ones_S1, x_SPxT], dim=-1)
        pairwise_features_Sff = torch.einsum("si,sj->sij", x_Sf, x_Sf)
        idx = torch.triu_indices(
            self.n_ohe_features, self.n_ohe_features, device=ohe_seq_SPT.device
        )
        x_SF = pairwise_features_Sff[:, idx[0], idx[1]]
        y_SO = self.linear_layer(x_SF)
        return y_SO


def pca_embed_init(
    pretrained_weights: torch.Tensor,
    pretrained_vocab: dict[str, int],
    target_vocab: dict[str, int],
    n_components: int,
    target_vocab_size: Optional[int] = None,
) -> torch.Tensor:
    """Project pretrained embeddings onto their top principal components, mapped to target vocab indices.

    Finds shared tokens between the pretrained and target vocabularies,
    computes PCA over the pretrained embeddings of those shared tokens,
    and returns the projected embeddings arranged in the target vocabulary's
    index order.

    Args:
        pretrained_weights: Embedding weight matrix, shape (V_pretrained, D_pretrained).
        pretrained_vocab: Token string → index mapping for the pretrained model.
        target_vocab: Token string → index mapping for the target model.
        n_components: Number of principal components to keep.
        target_vocab_size: Total number of rows in the output tensor. Must be
            large enough to contain all indices in ``target_vocab``.  When
            ``None`` (default), inferred as ``max(target_vocab.values()) + 1``.
            Set this when the target embedding table has extra slots (e.g. a
            padding index) that are not listed in ``target_vocab``.

    Returns:
        Tensor of shape (target_vocab_size, n_components) with PCA-projected
        embeddings at the correct target indices. Rows for tokens not found in
        the pretrained vocabulary (or extra slots) are zero.

    Raises:
        ValueError: If no shared tokens exist, n_components exceeds the
            number of shared tokens or the pretrained embedding dimension,
            or target_vocab_size is too small for the target vocab.
    """
    shared_tokens = sorted(set(pretrained_vocab.keys()) & set(target_vocab.keys()))
    if len(shared_tokens) == 0:
        raise ValueError("No shared tokens between pretrained and target vocabularies")
    if n_components > len(shared_tokens):
        raise ValueError(
            f"n_components ({n_components}) exceeds number of shared tokens ({len(shared_tokens)})"
        )
    if n_components > pretrained_weights.shape[1]:
        raise ValueError(
            f"n_components ({n_components}) exceeds pretrained embedding dim ({pretrained_weights.shape[1]})"
        )

    min_target_size = max(target_vocab.values()) + 1
    if target_vocab_size is None:
        target_vocab_size = min_target_size
    elif target_vocab_size < min_target_size:
        raise ValueError(
            f"target_vocab_size ({target_vocab_size}) is too small to contain "
            f"all target vocab indices (max index = {min_target_size - 1})"
        )

    # Extract pretrained embeddings for shared tokens
    pretrained_indices = [pretrained_vocab[t] for t in shared_tokens]
    shared_embeddings = pretrained_weights[pretrained_indices].float()  # (n_shared, D)

    # PCA: center, SVD, project onto top-k components
    mean = shared_embeddings.mean(dim=0)
    centered = shared_embeddings - mean
    _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
    projected = centered @ Vt[:n_components].T  # (n_shared, n_components)

    # Place projections at the correct target indices
    output = torch.zeros(target_vocab_size, n_components)
    for i, token in enumerate(shared_tokens):
        output[target_vocab[token]] = projected[i]

    return output


class EmbeddingMLP(PredictiveModel, ABC):
    """MLP operating on learned token embeddings.

    Receives OHE input from PredictiveModel.get_log_probs, multiplies through
    a learned embedding matrix (``ohe @ embed.weight``), flattens across
    all positions, and passes through an MLP. The matmul embedding lookup
    is differentiable, enabling TAG gradient flow.

    Use :meth:`init_embed_from_pretrained_pca` to initialize the embedding
    layer from a pretrained model's embeddings (e.g. ESMC), compressed to
    ``embed_dim`` principal components with automatic cross-tokenizer mapping.

    Tensor Dimension Labels:
        S: batch (sample) index
        P: position in sequence
        E: embedding dimension
        O: output dimension
    """

    def __init__(
        self,
        tokenizer,
        sequence_length: int,
        embed_dim: int,
        model_dim: int,
        n_layers: int,
        output_dim: int,
        padding_idx: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__(tokenizer)
        self.vocab_size = tokenizer.vocab_size
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.model_dim = model_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.padding_idx = (
            padding_idx if padding_idx is not None else tokenizer.pad_token_id
        )
        self.dropout = dropout

        self.embed = nn.Embedding(
            self.vocab_size,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )

        layers: list[nn.Module] = [
            nn.Linear(self.sequence_length * self.embed_dim, self.model_dim)
        ]
        for _ in range(n_layers - 1):
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(self.model_dim, self.model_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(self.model_dim, self.output_dim))
        self.layers = nn.Sequential(*layers)

    def init_embed_from_pretrained_pca(
        self,
        source: nn.Embedding,
        source_vocab: dict[str, int],
        target_vocab: dict[str, int],
    ) -> None:
        """Initialize embedding layer from PCA of a pretrained model's embeddings.

        Finds tokens shared between source and target vocabularies, runs PCA
        on their pretrained embeddings, and copies the first ``self.embed_dim``
        principal components into this model's embedding layer at the correct
        target indices.  Unmatched rows and the padding row are zeroed.

        Args:
            source: Pretrained embedding layer (e.g. ``esmc_model.embed``).
            source_vocab: Token string → index mapping for the pretrained model
                (e.g. ``esm_tokenizer.vocab``).
            target_vocab: Token string → index mapping for this model
                (e.g. ``mpnn_tokenizer.vocab``).
        """
        weights = pca_embed_init(
            pretrained_weights=source.weight.detach(),
            pretrained_vocab=source_vocab,
            target_vocab=target_vocab,
            n_components=self.embed_dim,
            target_vocab_size=self.vocab_size,
        )
        self.embed.weight.data.copy_(weights)
        if self.padding_idx is not None:
            self.embed.weight.data[self.padding_idx].zero_()

    def forward(self, ohe_seq_SPT: torch.FloatTensor) -> torch.FloatTensor:
        x_SPE = ohe_seq_SPT @ self.embed.weight  # differentiable embedding lookup
        x_SPxE = x_SPE.reshape(x_SPE.size(0), -1)
        return self.layers(x_SPxE)
