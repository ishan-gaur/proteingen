import torch
from torch import nn
from torch.nn import functional as F
from typing import Callable, Any, Dict, Optional
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dfm.probability_model import ProbabilityModel
from transformers import PreTrainedTokenizerBase


class PredictiveModel(ProbabilityModel, ABC):
    """Base class for predictive models used in guidance.

    Subclasses implement ``forward()`` to return logits by wrapping a base
    predictor model and converting its raw output to logits (via a
    raw_output_to_logits conversion). ``ProbabilityModel.get_log_probs()``
    then handles temperature-scaled log_softmax.

    Child classes must set ``self.input_dim`` (vocab size) for the
    ``target_log_probs_given_seq`` convenience method to work.
    """

    def __init__(self, model: nn.Module, tokenizer: PreTrainedTokenizerBase):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def forward(
        self, ohe_seq_SPT: torch.FloatTensor, **kwargs
    ):  # note this is a float and one-hot encoded as we might want to take a gradient on it
        raw_output = self.model(ohe_seq_SPT)
        return raw_output

    @abstractmethod
    def format_raw_to_logits(
        self, raw_output: Any, ohe_seq_SPT: torch.FloatTensor, **kwargs
    ):
        # sets a specification of the target so that the raw forward regression
        # value or class logits turn into a cdf or class probability
        ...

    # TODO[pi] make a note in the documentation somewhere that we recommend
    # making a typeddict for the conditioning information and setting
    # the kwargs for these functions as regular arguments
    @abstractmethod
    def preprocess_observations(
        self, observations: Dict[str, Any]
    ) -> Dict[str, Any]: ...

    def get_log_prob_target_from_seq(self, seq_SP):
        ohe_seq_SPT = F.one_hot(seq_SP, self.input_dim).float()
        return self.get_log_probs(ohe_seq_SPT)


class RealValuedPredictiveModel(PredictiveModel, ABC):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.threshold = None

    @contextmanager
    def with_target(self, threshold: Callable[torch.Tensor, bool]):
        pre_context_target = self.threshold
        self.threshold = threshold
        try:
            yield self
        finally:
            self.threshold = pre_context_target

    @abstractmethod
    def compute_log_cdf(
        self, prediction: torch.Tensor
    ) -> float:  # Prediction could be ensemble of values, mean/variance, monte-carlo samples, even the CDF
        pass

    def target_log_probs_given_ohe(self, ohe_seq_SPT):
        prediction = self.forward(ohe_seq_SPT)
        logp_y_g_x_S = self.compute_log_cdf(prediction)
        return logp_y_g_x_S


class EnsemblePredictiveModel(RealValuedPredictiveModel):
    pass


class GaussianPredictiveModel(RealValuedPredictiveModel):
    pass


class BinaryVariablePredictiveModel(ProbabilityModel):
    pass


class CategoricalVariablePredictiveModel(ProbabilityModel, ABC):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.target_class = None

    @contextmanager
    def with_target(self, target_class: int):
        pre_context_class = self.target_class
        self.target_class = target_class
        try:
            yield self
        finally:
            self.target_class = pre_context_class

    def target_log_probs_given_ohe(self, ohe_seq_SP: torch.LongTensor):
        logits_y_g_x_SC = self.forward(ohe_seq_SP)
        logp_y_g_x_S = F.log_softmax(logits_y_g_x_SC / self.temp)[:, self.target_class]
        return logp_y_g_x_S


# TODO[pi] RealValuedPredictiveModel, ClassValuedPredictiveModel, and
# EnsemblePredictiveModel should be refactored into TargetProbabilityMixin
# variants. TargetProbabilityMixin is an ABC whose concrete versions
# (ClassTargetMixin, RealValuedTargetMixin, EnsembleTargetMixin,
# GaussianTargetMixin) each define how a model's raw output (logits,
# regression value, ensemble, mean/variance) gets converted into
# log p(target_event | x). These would be mixed into PredictiveModel
# subclasses instead of baked into the class hierarchy.


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
    ) -> (torch.FloatTensor, torch.FloatTensor):
        # returns both the log probs and embeddings
        pass


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
    ):
        super().__init__()
        self.embed_model = embed_model
        self.embedding_dim = embed_model.EMB_DIM
        self.output_dim = output_dim
        self.w = nn.Linear(self.embedding_dim, self.output_dim)

    def forward(self, ohe_x_SPT: torch.LongTensor):
        x_ID = self.embed_model(ohe_x_SPT)
        y_IO = self.w(x_ID)
        return y_IO


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
