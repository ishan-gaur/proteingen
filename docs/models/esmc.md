# ESMC

ESM-C masked language model, available in 300m and 600m parameter variants. Wraps the ESM model as a `GenerativeModelWithEmbedding`, providing both generative sampling and differentiable embedding extraction (used by `LinearProbe` and TAG).

- **Output dim**: 64 (33 real vocab + 31 alignment padding — handled automatically by `MaskedModelLogitFormatter`)
- **Embedding dim**: 960 (300m) — set dynamically from model weights
- **LoRA support**: yes, via `apply_lora()`

```python
from protstar.models import ESMC

model = ESMC("esmc_300m").cuda()  # or "esmc_600m"
```
