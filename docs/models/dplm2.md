# DPLM-2

ByteDance's discrete diffusion protein language model ([DPLM-2](https://arxiv.org/abs/2410.13782), ICLR'25). Uses masked diffusion over an extended vocabulary that includes both amino acid and structure codebook tokens. Currently supports sequence-only mode.

- **Output dim**: 8229 (33 AA tokens + 8196 structure tokens — formatted by `MaskedModelLogitFormatter` to expose only the 20 standard AAs + mask)
- **Embedding dim**: 640 (150m), 1280 (650m), 2560 (3b) — set dynamically from model weights
- **LoRA support**: yes, via `apply_lora()`
- **Structure conditioning**: not yet supported (joint sequence+structure generation requires upstream's structure VQ-VAE tokenizer)

## Available checkpoints

HuggingFace hub (`airkingbd/`):

| Checkpoint | Params | Hidden | Layers |
|---|---|---|---|
| `airkingbd/dplm2_150m` | 150M | 640 | 30 |
| `airkingbd/dplm2_650m` | 650M | 1280 | 33 |
| `airkingbd/dplm2_3b` | 3B | 2560 | 36 |

```python
from protstar.models import DPLM2

model = DPLM2("airkingbd/dplm2_650m").cuda()  # default checkpoint
log_probs = model.get_log_probs_from_string(["ACDEFGHIK"])
```

DPLM-2 works with the same sampling, guidance, and probe infrastructure as the ESM models:

```python
from protstar.sampling import sample_ctmc_linear_interpolation

init_tokens = model.tokenizer(["<mask>" * 100], return_tensors="pt")["input_ids"].cuda()
sequences = sample_ctmc_linear_interpolation(model, init_tokens, n_steps=100)
```

!!! warning "Untied embedding weights"
    The HuggingFace config for DPLM-2 incorrectly sets `tie_word_embeddings=True`. The `DPLM2` wrapper overrides this to `False` before loading — if you load the model manually via `AutoModelForMaskedLM`, you'll get wrong logits.
