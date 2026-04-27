"""Backward-compatible model namespace.

New code should prefer ``proteingen.modeling``.
This module keeps old import styles working, including:

- ``from proteingen.models import ESMC``
- ``from proteingen.models import esmc`` (module)
- ``from proteingen.models.esm import ESMC``
"""

from __future__ import annotations

import sys
from importlib import import_module

from proteingen.modeling.models import (
    DPLM2,
    DPLM2Tokenizer,
    ESM3,
    ESM3IF,
    ESMC,
    ESMForgeAPI,
    Frame2seq,
    Frame2seqConditioning,
)

__all__ = [
    "ESMC",
    "ESM3",
    "ESM3IF",
    "ESMForgeAPI",
    "DPLM2",
    "DPLM2Tokenizer",
    "Frame2seq",
    "Frame2seqConditioning",
]

# Optional model classes
try:
    from proteingen.modeling.models import ProteinMPNN

    __all__.append("ProteinMPNN")
except ImportError:
    pass

try:
    from proteingen.modeling.models import PreTrainedStabilityPredictor

    __all__.append("PreTrainedStabilityPredictor")
except ImportError:
    pass


# Backward-compatible module aliases, e.g. "from proteingen.models import esmc"
_MODULE_ALIASES = {
    "esm": "proteingen.modeling.models.esm",
    "esmc": "proteingen.modeling.models.esm.esmc",
    "esm3": "proteingen.modeling.models.esm.esm3",
    "esm3if": "proteingen.modeling.models.esm.esm3if",
    "esm_api": "proteingen.modeling.models.esm.esm_api",
    "dplm2": "proteingen.modeling.models.dplm2",
    "frame2seq": "proteingen.modeling.models.frame2seq",
    "mpnn": "proteingen.modeling.models.mpnn",
    "rocklin_ddg": "proteingen.modeling.models.rocklin_ddg",
    "utils": "proteingen.data.structure",
}

for alias, target in _MODULE_ALIASES.items():
    try:
        module = import_module(target)
    except ImportError:
        continue
    globals()[alias] = module
    __all__.append(alias)
    sys.modules[f"{__name__}.{alias}"] = module
