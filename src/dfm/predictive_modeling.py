import torch
from torch import nn
from torch.nn import functional as F
from typing import Callable, Any, Dict
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


class EmbeddingMLP(nn.Module):
    """
    MLP operating on learned token embeddings.

    Each token is mapped to a learned embedding vector via ``nn.Embedding``,
    the embeddings are flattened across all positions, then passed through an MLP.
    This is a learned alternative to ``OneHotMLP``'s frozen identity embedding.

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

    def forward(self, x_SP: torch.LongTensor):
        x_SPE = self.embed(x_SP)
        x_SPxE = x_SPE.reshape(x_SPE.size(0), -1)
        y_hat_SO = self.layers(x_SPxE)
        return y_hat_SO
