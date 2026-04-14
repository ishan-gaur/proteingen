"""Modeling API: probabilistic models, guidance, and model wrappers."""

from .probability_model import ProbabilityModel
from .generative_modeling import (
    GenerativeModel,
    GenerativeModelWithEmbedding,
    TransitionFunc,
    LogitFormatter,
    PassThroughLogitFormatter,
    MaskedModelLogitFormatter,
    MPNNTokenizer,
)
from .predictive_modeling import (
    PredictiveModel,
    categorical_binary_logits,
    binary_logits,
    point_estimate_binary_logits,
    gaussian_binary_logits,
    LinearProbe,
    OneHotMLP,
    EmbeddingMLP,
    PairwiseLinearModel,
    pca_embed_init,
)
from .guide import (
    PreparedGuidanceInput,
    GuidanceProjection,
    LinearGuidanceProjection,
    TAG,
    DEG,
)
from .models import (
    ESMC,
    ESM3,
    ESM3IF,
    ESMForgeAPI,
    DPLM2,
    DPLM2Tokenizer,
)

__all__ = [
    "ProbabilityModel",
    "GenerativeModel",
    "GenerativeModelWithEmbedding",
    "TransitionFunc",
    "LogitFormatter",
    "PassThroughLogitFormatter",
    "MaskedModelLogitFormatter",
    "MPNNTokenizer",
    "PredictiveModel",
    "categorical_binary_logits",
    "binary_logits",
    "point_estimate_binary_logits",
    "gaussian_binary_logits",
    "LinearProbe",
    "OneHotMLP",
    "EmbeddingMLP",
    "PairwiseLinearModel",
    "pca_embed_init",
    "PreparedGuidanceInput",
    "GuidanceProjection",
    "LinearGuidanceProjection",
    "TAG",
    "DEG",
    "ESMC",
    "ESM3",
    "ESM3IF",
    "ESMForgeAPI",
    "DPLM2",
    "DPLM2Tokenizer",
]

try:
    from .models import ProteinMPNN

    __all__.append("ProteinMPNN")
except ImportError:
    pass

try:
    from .models import PreTrainedStabilityPredictor

    __all__.append("PreTrainedStabilityPredictor")
except ImportError:
    pass
