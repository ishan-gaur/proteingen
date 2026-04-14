"""Backward-compatible model namespace.

New code should prefer ``protstar.modeling``.
This module keeps old import styles working, including:

- ``from protstar.models import ESMC``
- ``from protstar.models import esmc`` (module)
- ``from protstar.models.esm import ESMC``
"""

from __future__ import annotations

import sys
from importlib import import_module

from protstar.modeling.models import (
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
    from protstar.modeling.models import ProteinMPNN

    __all__.append("ProteinMPNN")
except ImportError:
    pass

try:
    from protstar.modeling.models import PreTrainedStabilityPredictor

    __all__.append("PreTrainedStabilityPredictor")
except ImportError:
    pass


# Backward-compatible module aliases, e.g. "from protstar.models import esmc"
_MODULE_ALIASES = {
    "esm": "protstar.modeling.models.esm",
    "esmc": "protstar.modeling.models.esm.esmc",
    "esm3": "protstar.modeling.models.esm.esm3",
    "esm3if": "protstar.modeling.models.esm.esm3if",
    "esm_api": "protstar.modeling.models.esm.esm_api",
    "dplm2": "protstar.modeling.models.dplm2",
    "frame2seq": "protstar.modeling.models.frame2seq",
    "mpnn": "protstar.modeling.models.mpnn",
    "rocklin_ddg": "protstar.modeling.models.rocklin_ddg",
    "utils": "protstar.data.structure",
}

for alias, target in _MODULE_ALIASES.items():
    try:
        module = import_module(target)
    except ImportError:
        continue
    globals()[alias] = module
    __all__.append(alias)
    sys.modules[f"{__name__}.{alias}"] = module
