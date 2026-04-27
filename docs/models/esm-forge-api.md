# ESM Forge API

Remote inference via the [EvolutionaryScale Forge](https://forge.evolutionaryscale.ai) API. Wraps Forge clients to provide the same `get_log_probs` interface as local ESM models — no local weights needed. Automatically selects the right client (ESM3 vs ESMC) based on model name.

- **Output dim**: 64 (same tokenizer as local ESM models)
- **Available models**: `esmc-6b-2024-12`, `esm3-open-2024-03`, and others on the Forge platform
- **Structure conditioning**: ESM3 models only (remote VQ-VAE encoding via `preprocess_observations`)
- **LoRA / fine-tuning**: not supported (remote inference)
- **Gradients**: not available (no `embed`, no TAG guidance)

```python
import os
from proteingen.models import ESMForgeAPI

# ESMC — no conditioning, just masked LM
model = ESMForgeAPI("esmc-6b-2024-12", token=os.environ["FORGE_TOKEN"])
log_probs = model.get_log_probs_from_string(["ACDEFGHIK"])
```

## Structure conditioning (ESM3 only)

For ESM3 models, structure conditioning works through the same `preprocess_observations` / `set_condition_` interface as the local ESM3 wrapper. The VQ-VAE encoding happens remotely:

```python
model = ESMForgeAPI("esm3-open-2024-03", token=os.environ["FORGE_TOKEN"])
coords = ...  # atom37 format, shape (L, 37, 3)

model.set_condition_({"coords_RAX": coords})
log_probs = model.get_log_probs_from_string(["ACDEFGHIK" + "A" * (L - 9)])
```

!!! note "When to use Forge vs local models"
    Forge is useful for accessing larger models (e.g. ESMC-6B) that don't fit in local GPU memory, or for quick experiments without downloading weights. For training, LoRA, TAG guidance, or any workflow requiring gradients, use the local model wrappers instead.
