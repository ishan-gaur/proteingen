# ProGen3

Profluent's autoregressive protein language model wrapper (`proteingen.models.ProGen3`).

Unlike masked models (ESMC, ESM3, DPLM-2), ProGen3 generates left-to-right (N→C terminal):

- **Output dim**: 134
- **Embedding dim**: checkpoint-dependent (`EMB_DIM` read from weights)
- **Primary use cases**: autoregressive generation, sequence scoring, embedding extraction

## Available checkpoints

| Checkpoint | Params |
|---|---:|
| `Profluent-Bio/progen3-112m` | 112M |
| `Profluent-Bio/progen3-219m` | 219M |
| `Profluent-Bio/progen3-339m` | 339M |
| `Profluent-Bio/progen3-762m` | 762M |
| `Profluent-Bio/progen3-1b` | 1B |
| `Profluent-Bio/progen3-3b` | 3B |

```python
from proteingen.models import ProGen3

model = ProGen3("Profluent-Bio/progen3-112m").cuda()
```

## Generation APIs

### Fixed-length generation via `sample()`

```python
from proteingen.sampling import sample

init_x = ["<mask>" * 100 for _ in range(4)]
traj = sample(model, init_x, in_order="left_to_right")
print(traj["sequences"])
```

### Open-ended generation via `generate()`

```python
result = model.generate(n=4, max_new_tokens=256, temperature=0.8, top_p=0.95)
print(result["sequences"])
```

### Sequence scoring

```python
scores = model.score(["ACDEFGHIKLMNPQRSTVWY"])
print(scores["log_likelihood"], scores["perplexity"])
```

!!! note "Dependency caveat"
    The upstream `progen3` package may need additional runtime deps (not declared in package metadata), notably `megablocks`/`grouped_gemm` (and often Flash Attention) for MoE checkpoints.

See also: [Autoregressive Generation (ProGen3)](../examples/autoregressive-generation.md)
