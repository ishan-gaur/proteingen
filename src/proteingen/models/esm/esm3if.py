"""Deprecated — use ESM3 with structure conditioning instead.

ESM3IF was an early inverse-folding wrapper around ESM3. ESM3 now fully
supports optional structure conditioning via ``set_condition_()`` /
``conditioned_on()``, making this class redundant.

Migration::

    # Old
    model = ESM3IF().cuda()
    model.set_condition_({"coords_RAX": coords})

    # New
    from proteingen.models import ESM3
    model = ESM3().cuda()
    model.set_condition_({"coords_RAX": coords})
"""

from __future__ import annotations

import warnings

from proteingen.models.esm.esm3 import ESM3


class ESM3IF(ESM3):
    """Deprecated: use ``ESM3`` with ``set_condition_()`` instead.

    This thin subclass exists only for backwards compatibility. It issues a
    deprecation warning on construction and delegates everything to ``ESM3``.
    """

    def __init__(self, esm3_checkpoint: str = "esm3-open", **kwargs):
        warnings.warn(
            "ESM3IF is deprecated — use ESM3 with set_condition_() or "
            "conditioned_on() for structure-conditioned generation. "
            "ESM3IF will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(esm3_checkpoint=esm3_checkpoint, **kwargs)
