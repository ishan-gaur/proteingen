# Design Philosophy

ProteinGen is built around a small number of composable abstractions that mirror the math of guided generation. Understanding these three base classes is all you need to use the library.

## The three base classes

### ProbabilityModel

Everything in ProteinGen is a `ProbabilityModel` — an `nn.Module` that produces log-probability distributions. This shared base class provides:

- **Temperature** — scale the sharpness of distributions
- **Conditioning** — attach observations (e.g., structure coordinates) that the model conditions on
- **Log-prob pipeline** — `get_log_probs(x)` chains collation → forward → logit formatting → log-softmax

Two abstract methods must be implemented by subclasses:

- `forward(x, **kwargs)` — run the model, return raw output (can be any type)
- `format_raw_to_logits(raw_output, x, **kwargs)` — extract a float tensor of logits from raw output

#### Temperature

All three styles — in-place, chained, and context-managed — are available:

```python
model.set_temp_(2.0)                # in-place mutation
model = model.set_temp(2.0)         # returns self (chainable)
with model.with_temp(2.0):          # reverts when exiting
    log_probs = model.get_log_probs(x)
```

#### Conditioning

Attach observations that persist across calls:

```python
# In-place
model.set_condition_({"coords_RAX": coords})

# Context manager (reverts on exit)
with model.conditioned_on({"coords_RAX": coords}):
    log_probs = model.get_log_probs(x)
```

`preprocess_observations` runs once when conditions are set (e.g., encoding a structure). `collate_observations` tiles observations to match batch size at inference time.

#### Checkpointing

Save and restore models with their constructor arguments:

```python
model.save("checkpoints/my_model")
restored = MyModel.from_checkpoint("checkpoints/my_model")
```

Subclasses implement `_save_args()` to return JSON-serializable constructor kwargs.

---

### GenerativeModel

A **concrete** `ProbabilityModel` subclass that wraps any `nn.Module` generative model via composition:

```python
from proteingen import GenerativeModel, MaskedModelLogitFormatter

model = GenerativeModel(
    model=my_nn_module,
    tokenizer=my_tokenizer,
    logit_formatter=MaskedModelLogitFormatter(my_tokenizer, output_dim=64),
)
```

- `forward` delegates to `self.model(seq, **kwargs)`
- `format_raw_to_logits` applies the logit formatter
- Override `format_raw_to_logits` when the wrapped model returns non-tensor output (e.g., ESM dataclasses)

**LoRA support** is built in: `apply_lora()`, `save_lora()`, `load_lora()`.

#### GenerativeModelWithEmbedding

An ABC extending `GenerativeModel` for models that support differentiable embedding extraction. Subclasses implement:

- `differentiable_embedding(ohe) → embeddings` — OHE through embedding layer + transformer
- `embedding_to_outputs(embeddings) → raw_output` — embeddings through the output head

This enables `LinearProbe` to extract and cache embeddings, and provides a differentiable path from one-hot inputs through the full model (needed for TAG gradients).

---

### PredictiveModel

An ABC extending `ProbabilityModel` for models that answer "what is log p(target | sequence)?". Uses a **binary logit pattern**: `format_raw_to_logits` returns `(B, 2)` logits `[false_logit, true_logit]`, and `get_log_probs` extracts the `true_logit` after log-softmax.

```python
class MyPredictor(PredictiveModel):
    def forward(self, ohe, **kwargs):
        return self.mlp(ohe.flatten(1))  # raw scalar predictions

    def format_raw_to_logits(self, raw, ohe, **kwargs):
        return point_estimate_binary_logits(raw.squeeze(-1), threshold=0.7, k=10)
```

#### Four layers

A predictive model integration decomposes into four separable layers. Understanding this decomposition makes it clear what you're building vs reusing:

1. **Raw Predictor** — the original pretrained model (architecture + weights), ported with minimal changes. Not proteingen-specific.
2. **Binary Logit Function** — converts raw output to `(B, 2)` binary logits. Independent of the model — the same predictor could use different functions. The library provides `binary_logits`, `categorical_binary_logits`, `point_estimate_binary_logits`, and `gaussian_binary_logits`.
3. **Template Model Class** *(optional)* — a reusable architecture pattern (e.g. `LinearProbe`, `EmbeddingMLP`). If the predictor's architecture generalizes, add a template. If it's one-off, subclass `PredictiveModel` directly.
4. **PredictiveModel Subclass** — thin glue wiring 1–3 together with conditioning, tokenizer, and OHE basis. If the other layers are well-designed, this should be mostly boilerplate.

See the [contributing guide](../contributing.md#the-four-layers) for details on each layer.

#### Target management

```python
model.set_target_(True)               # in-place
with model.with_target(True):         # context manager
    log_prob = model.get_log_probs(x)  # log p(target=True | x)
```

#### Gradient access (for TAG)

```python
grad = model.grad_log_prob(seq_SP)  # ∂log p(target|x) / ∂OHE, shape (B, L, K)
```

#### Template subclasses

- **`LinearProbe`** — frozen `GenerativeModelWithEmbedding` + `nn.Linear` head
- **`EmbeddingMLP`** — learnable embeddings + MLP, with PCA initialization from pretrained models
- **`OneHotMLP`** — flattened one-hot + MLP

All are ABCs — you implement `format_raw_to_logits` using the binary logit functions listed above.

---

## Guidance

`TAG` (Taylor-Approximate Guidance) and `DEG` (Discrete Enumeration Guidance) combine a generative model with a predictive model using Bayes' rule:

$$
p_\text{guided}(x_t | x_{<t}) \propto p_\text{gen}(x_t | x_{<t}) \cdot p_\text{pred}(\text{target} | x)^\gamma
$$

Both are `GenerativeModel` subclasses — they produce guided log-probs that can be passed directly to any sampler.

- **TAG** uses first-order Taylor expansion of the predictive model's log-prob. Works well when gradients are reliable.
- **DEG** enumerates all 20 amino acids at each position and reweights. More robust for frozen-LM probes where gradients through the transformer are unreliable.

`GuidanceProjection` handles cross-tokenizer mapping when the predictive and generative models use different vocabularies.

---

## Sampling

`sample_any_order` generates sequences by unmasking one position at a time in random order, using `model.get_log_probs` at each step:

```python
from proteingen import sample_any_order
from proteingen.models import ESMC

model = ESMC().cuda()
sequences = sample_any_order(model, ["<mask>" * 100] * 8)
```

### Linear interpolation sampler

`sample_linear_interpolation` generates sequences by interpolating between the current token distribution and the model's predicted distribution over a fixed number of steps. At each step $i$ of $N$ total:

$$
p_\text{next}(x) = \frac{N - i - 1}{N - i} \cdot \mathbb{1}[x = x_\text{current}] + \frac{1}{N - i} \cdot p_\text{model}(x)
$$

Tokens are resampled from this mixture at every position simultaneously, so the distribution gradually shifts from the initial state (fully masked) to the model's predicted distribution. Unlike ancestral sampling which unmasks one position at a time, linear interpolation updates all positions in parallel at each step.

```python
from proteingen.sampling import sample_linear_interpolation
from proteingen.models import ESMC

model = ESMC().cuda()
sequences = sample_linear_interpolation(model, ["<mask>" * 100] * 8, n_steps=50)
```

### Flow-matching Euler sampler

`sample_flow_matching_legacy` integrates a rate matrix using Euler steps, following the continuous-time flow-matching framework. At each time step, the model predicts an $x_1$ distribution, and a rate matrix $R_t$ is constructed such that masked positions transition toward the predicted distribution at a rate proportional to $1/(1-t)$. With optional stochasticity, unmasked positions can also remask.

When a predictive model is provided, guidance is applied by reweighting the rate matrix with likelihood ratios — either via enumeration (DEG-style) or first-order Taylor approximation (TAG-style).

```python
from proteingen.sampling import sample_flow_matching_legacy
from proteingen.models import ESMC

model = ESMC().cuda()
sequences = sample_flow_matching_legacy(model, ["<mask>" * 100] * 8, dt=0.01)
```

Key parameters:

- `dt` — step size (default 0.01, i.e. 100 steps)
- `x1_temp` — temperature applied to the model's $x_1$ prediction
- `stochasticity` — controls remasking rate (0 = deterministic flow, >0 = stochastic)
- `argmax_final` — if True, remaining masked positions are filled with argmax at $t=1$
- `predictor_log_prob` — optional guidance function (use `build_legacy_predictor_log_prob` to construct from a TAG model)

---

## Composition

The key design insight: because TAG, DEG, and all models share the `ProbabilityModel` interface, they compose naturally. You can layer multiple guidance signals, swap generative backbones, or mix sampling strategies without changing any code.
