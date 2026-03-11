import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dfm.probability_model import ProbabilityModel
from transformers import PreTrainedTokenizerBase


class PredictiveModel(ProbabilityModel, ABC):
    """Base class for predictive models used in guidance.

    Predictive models answer: "what is log p(target_event | sequence)?"

    The target event is set via ``set_target_()`` or the ``target()`` context
    manager. ``forward()`` returns raw predictions (class logits, regression
    values, etc.). ``format_raw_to_logits()`` converts those to binary logits
    ``(B, 2)``: ``[false_logit, true_logit]``. The inherited
    ``ProbabilityModel.get_log_probs`` applies temperature-scaled log_softmax,
    and this class's override takes ``[:, 1]`` to return the scalar
    log p(target=True | x).

    Pipeline::

        forward(ohe_seq) → raw output (class logits, regression value, ...)
            ↓
        format_raw_to_logits(raw) → (B, 2) binary logits [false, true]
            ↓
        ProbabilityModel.get_log_probs: log_softmax(logits / temp) → (B, 2)
            ↓
        PredictiveModel.get_log_probs: [:, 1] → (B,) log p(target | x)

    Subclasses must set ``self.input_dim`` (vocabulary size for OHE).
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__()
        self.tokenizer = tokenizer
        self._target = None


    def set_target_(self, target_spec):
        """Set the target event in-place."""
        self._target = target_spec

    def set_target(self, target_spec):
        """Set the target event, returning self for chaining."""
        self.set_target_(target_spec)
        return self

    @contextmanager
    def with_target(self, target_spec):
        """Context manager: temporarily set target, revert on exit."""
        old = self._target
        self.set_target_(target_spec)
        try:
            yield self
        finally:
            self._target = old

    @abstractmethod
    def forward(self, ohe_seq_SPT: torch.FloatTensor, **kwargs) -> Any:
        """Return raw predictions from one-hot encoded input.

        Must be differentiable w.r.t. ohe_seq_SPT for TAG guidance.
        """
        ...

    @abstractmethod
    def format_raw_to_logits(
        self, raw_output: Any, ohe_seq_SPT: torch.FloatTensor, **kwargs
    ) -> torch.FloatTensor:
        """Convert raw predictions → binary logits (B, 2): [false_logit, true_logit].

        Uses ``self._target`` to determine what event is being evaluated.
        The parent's ``get_log_probs`` applies ``log_softmax(logits / temp)``
        on top.
        """
        ...

    def get_log_probs(self, ohe_seq_SPT: torch.FloatTensor) -> torch.FloatTensor:
        """Return log p(target=True | x), scalar per sequence.

        Calls the parent pipeline (forward → format_raw_to_logits →
        log_softmax with temperature) to get (B, 2), then takes [:, 1].
        """
        assert self._target is not None, (
            "Target not set. Call set_target_() or use target() context manager."
        )
        log_probs_B2 = super().get_log_probs(ohe_seq_SPT)  # (B, 2)
        assert log_probs_B2.shape[1] == 2, (
            f"Expected binary logits (B, 2) from format_raw_to_logits, got shape {log_probs_B2.shape}"
        )
        return log_probs_B2[:, 1]  # (B,)

    def get_log_probs_from_string(self, seq_SP: torch.LongTensor) -> torch.FloatTensor:
        """Convenience: one-hot encode token IDs, then get_log_probs."""
        ohe_seq_SPT = F.one_hot(seq_SP, self.input_dim).float()
        return self.get_log_probs(ohe_seq_SPT)


class CategoricalPredictiveModel(PredictiveModel, ABC):
    """Predictive model with categorical (multi-class) output.

    Target is a class index (int). ``format_raw_to_logits`` converts
    multi-class logits ``(B, C)`` to binary logits ``(B, 2)`` by splitting
    the target class from the rest:

    - ``true_logit = logits[:, target_class]``
    - ``false_logit = logsumexp(logits for non-target classes)``

    Subclasses implement ``forward()`` to return class logits ``(B, C)``.

    Optionally pass ``class_names`` to enable setting target by string::

        model = MyClassifier(tokenizer, class_names={"stable": 0, "unstable": 1})
        model.set_target_("stable")  # equivalent to set_target_(0)
    """

    def __init__(
        self,
        tokenizer,
        class_names: Optional[Dict[str, int]] = None,
    ):
        super().__init__(tokenizer)
        self.class_names = class_names or {}

    def set_target_(self, target_spec):
        """Set target class. Accepts int (class index) or str (class name)."""
        if isinstance(target_spec, str):
            assert target_spec in self.class_names, (
                f"Unknown class name '{target_spec}'. "
                f"Known classes: {list(self.class_names.keys())}"
            )
            self._target = self.class_names[target_spec]
        else:
            self._target = target_spec

    def set_target(self, target_spec):
        self.set_target_(target_spec)
        return self

    def format_raw_to_logits(
        self, raw_output: torch.FloatTensor, ohe_seq_SPT: torch.FloatTensor, **kwargs
    ) -> torch.FloatTensor:
        logits_BC = raw_output
        target_logit_B = logits_BC[:, self._target]
        # logsumexp of non-target classes = logit for "not target"
        C = logits_BC.shape[-1]
        mask = torch.ones(C, dtype=torch.bool, device=logits_BC.device)
        mask[self._target] = False
        false_logit_B = torch.logsumexp(logits_BC[:, mask], dim=-1)
        return torch.stack([false_logit_B, target_logit_B], dim=-1)  # (B, 2)


class BinaryPredictiveModel(PredictiveModel, ABC):
    """Predictive model with a single binary logit output.

    ``forward()`` returns a single logit per sequence where
    ``sigmoid(logit) = P(positive | x)``. Uses the identity
    ``sigmoid(x) = softmax([0, x])[1]`` to produce binary logits.

    Target is a bool: ``True`` for P(positive), ``False`` for P(negative).
    Defaults to ``True``.
    """

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self._target = True

    def format_raw_to_logits(
        self, raw_output: torch.FloatTensor, ohe_seq_SPT: torch.FloatTensor, **kwargs
    ) -> torch.FloatTensor:
        logit_B = raw_output.reshape(-1)
        zero_B = torch.zeros_like(logit_B)
        if self._target:
            return torch.stack([zero_B, logit_B], dim=-1)  # (B, 2)
        else:
            return torch.stack([logit_B, zero_B], dim=-1)  # (B, 2)


class PointEstimatePredictiveModel(PredictiveModel, ABC):
    """Predictive model with a real-valued point estimate output.

    ``forward()`` returns a scalar prediction per sequence ``(B,)`` or ``(B, 1)``.
    Target is a threshold (float). Converts to binary logits using a steep
    sigmoid approximation of the step function.

    Not differentiable through the threshold — use DEG, not TAG.

    The sharpness parameter ``k`` controls how steep the step is.
    Large ``k`` → hard step (good for DEG). Smaller ``k`` → softer
    (but still not a proper probabilistic model — use
    ``GaussianPredictiveModel`` if you have uncertainty estimates).
    """

    def __init__(self, tokenizer, k: float = 100.0):
        super().__init__(tokenizer)
        self.k = k

    def format_raw_to_logits(
        self, raw_output: torch.FloatTensor, ohe_seq_SPT: torch.FloatTensor, **kwargs
    ) -> torch.FloatTensor:
        pred_B = raw_output.reshape(-1)
        # steep sigmoid approximation: sigmoid(k * (pred - threshold))
        # maps to binary logits via sigmoid(x) = softmax([0, x])[1]
        logit_B = self.k * (pred_B - self._target)
        zero_B = torch.zeros_like(logit_B)
        return torch.stack([zero_B, logit_B], dim=-1)  # (B, 2)


class GaussianPredictiveModel(PredictiveModel, ABC):
    """Predictive model with Gaussian (mean, variance) output.

    ``forward()`` returns ``(B, 2)`` where column 0 is the mean and
    column 1 is the log-variance. Target is a threshold (float).
    Converts to binary logits via the Gaussian CDF:
    ``P(Y > threshold) = 1 - Phi((threshold - mu) / sigma)``.

    Differentiable through both mean and variance — works with TAG.
    """

    def format_raw_to_logits(
        self, raw_output: torch.FloatTensor, ohe_seq_SPT: torch.FloatTensor, **kwargs
    ) -> torch.FloatTensor:
        mu_B = raw_output[:, 0]
        log_var_B = raw_output[:, 1]
        sigma_B = (log_var_B / 2).exp()

        # P(Y > threshold) = Phi((mu - threshold) / sigma)
        # We need unnormalized logits since the parent applies log_softmax.
        # Use log_ndtr for numerically stable log Phi(z):
        #   logit = log(p_above / p_below) = log_ndtr(z) - log_ndtr(-z)
        # Then softmax([0, logit])[1] = sigmoid(logit) = p_above.
        z_B = (mu_B - self._target) / sigma_B
        # log_ndtr handles extreme z values without NaN
        log_p_above = torch.special.log_ndtr(z_B)
        log_p_below = torch.special.log_ndtr(-z_B)
        logit_B = log_p_above - log_p_below  # log-odds

        zero_B = torch.zeros_like(logit_B)
        return torch.stack([zero_B, logit_B], dim=-1)  # (B, 2)
# ==========================================================================================
# ==========================================================================================
# The following classes are templates to train your own predictive models with.
# When creating your own predictive models, make sure that they inherit from PredictiveModel
# and wrap around one of the template models below. See the dfm/models/ folder for examples.
# ==========================================================================================
# ==========================================================================================


class PreTrainedEmbeddingModel(nn.Module, ABC):
    EMB_DIM = None

    @abstractmethod
    def forward_ohe(
        self, ohe_seq_SPT: torch.FloatTensor
    ) -> Any:
        """Differentiable forward pass from one-hot encoded input."""
        ...

    # @staticmethod
    # def masked_pool_embeddings(
    #     embeddings_SPD: torch.Tensor,
    #     seq_SP: torch.LongTensor,
    #     special_ids: set[int],
    # ) -> torch.Tensor:
    #     """Mean-pool embeddings over non-special positions. Shape (S, D)."""
    #     mask_SP = torch.ones_like(seq_SP, dtype=torch.bool)
    #     for sid in special_ids:
    #         mask_SP = mask_SP & (seq_SP != sid)
    #     mask_SP1 = mask_SP.unsqueeze(-1).float()
    #     return (embeddings_SPD * mask_SP1).sum(dim=1) / mask_SP1.sum(dim=1).clamp(min=1)


class LinearProbe(nn.Module):
    """
    Linear probe on top of pre-computed embeddings.

    Tensor Dimension Labels:
        I: batch index
        D: embedding dimension
        O: output dimension
    """

    def __init__(
        self,
        embed_model: PreTrainedEmbeddingModel,
        output_dim: int,
        pooling_fn: Optional[callable] = None,
    ):
        super().__init__()
        self.embed_model = embed_model
        self.embedding_dim = embed_model.EMB_DIM
        self.output_dim = output_dim
        self.w = nn.Linear(self.embedding_dim, self.output_dim)
        self.pooling_fn = pooling_fn or (lambda emb_SPD, seq_SP: emb_SPD.mean(dim=1))

        for p in self.embed_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def compute_embeddings(
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
            token_ids = tokenizer(
                batch_seqs, padding=True, return_tensors="pt"
            )["input_ids"].to(device)
            output = self.embed_model(token_ids)
            pooled = self.pooling_fn(output.embeddings, token_ids)
            all_embeddings.append(pooled.cpu())
            if (start // batch_size) % 50 == 0:
                print(f"  Embedded {min(start + batch_size, n):>6d} / {n}")
        return torch.cat(all_embeddings, dim=0)

    def forward(self, seq_SP: torch.LongTensor):
        output = self.embed_model(seq_SP)
        pooled_SD = self.pooling_fn(output.embeddings, seq_SP)
        return self.w(pooled_SD)


class OneHotMLP(nn.Module):
    """
    MLP operating on one-hot encoded sequences.

    Uses a frozen identity embedding to convert token indices to one-hot vectors,
    flattens across all positions, then passes through an MLP.

    Tensor Dimension Labels:
        S: batch (sample) index
        P: position in sequence
        T: token dimension (one-hot)
        O: output dimension
    """

    def __init__(
        self,
        vocab_size: int,
        sequence_length: int,
        model_dim: int,
        n_layers: int,
        output_dim: int,
        padding_idx: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.model_dim = model_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.padding_idx = padding_idx
        self.dropout = dropout

        # Frozen one-hot embedding: each token maps to its one-hot vector
        self.embed = nn.Embedding(
            self.vocab_size,
            self.vocab_size,
            self.padding_idx,
            _weight=torch.eye(self.vocab_size),
            _freeze=True,
        )

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

    def forward(self, x_SP: torch.LongTensor):
        x_SPT = F.one_hot(x_SP, num_classes=self.vocab_size).float()
        x_SPxT = x_SPT.reshape(x_SPT.size(0), -1)
        y_hat_SO = self.layers(x_SPxT)
        return y_hat_SO


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
        raise ValueError(
            "No shared tokens between pretrained and target vocabularies"
        )
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


class EmbeddingMLP(nn.Module):
    """
    MLP operating on learned token embeddings.

    Each token is mapped to a learned embedding vector via ``nn.Embedding``,
    the embeddings are flattened across all positions, then passed through an MLP.
    This is a learned alternative to ``OneHotMLP``'s frozen identity embedding.

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
        vocab_size: int,
        sequence_length: int,
        embed_dim: int,
        model_dim: int,
        n_layers: int,
        output_dim: int,
        padding_idx: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.model_dim = model_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.padding_idx = padding_idx
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
        self.embed.weight.data[self.padding_idx].zero_()

    def forward(self, x_SP: torch.LongTensor):
        x_SPE = self.embed(x_SP)
        x_SPxE = x_SPE.reshape(x_SPE.size(0), -1)
        y_hat_SO = self.layers(x_SPxE)
        return y_hat_SO
