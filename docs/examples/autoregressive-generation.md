# Autoregressive Generation (ProGen3)

Generate novel protein sequences using ProGen3's autoregressive (left-to-right) sampling. Unlike masked models that iteratively unmask positions, ProGen3 generates one amino acid at a time from N-terminal to C-terminal.

## Quick Start

```python
from proteingen.models import ProGen3
from proteingen import sample

model = ProGen3("Profluent-Bio/progen3-112m").cuda()

# Fixed-length: 5 proteins of 100 residues each
init_x = ["<mask>" * 100 for _ in range(5)]
result = sample(model, init_x, in_order="left_to_right")
print(result["sequences"])
```

```bash
uv run python examples/autoregressive_generation.py
```

## Fixed-Length Generation via `sample()`

Use `<mask>` tokens as placeholders and `sample()` fills them left-to-right:

```python
from proteingen.models import ProGen3
from proteingen import sample

model = ProGen3("Profluent-Bio/progen3-112m").cuda()

init_x = ["<mask>" * 200 for _ in range(10)]
result = sample(model, init_x, in_order="left_to_right")

for i, seq in enumerate(result["sequences"]):
    print(f"Protein {i}: {len(seq)} residues")
    print(f"  {seq[:80]}...")
```

This uses the same `sample()` interface as masked models — the `AutoregressiveLogitFormatter` ensures only the next unfilled position gets non-trivial logits at each step.

## Open-Ended Generation (Variable Length)

Let the model decide when to stop:

```python
result = model.generate(
    n=10,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.95,
)
for i, seq in enumerate(result["sequences"]):
    print(f"Protein {i}: {len(seq)} residues")
```

## Prompted Generation (Sequence Completion)

Provide an N-terminal prefix and let the model complete the protein:

```python
result = model.generate(
    prompt="MKTLLLTLVVVTIVCLD",
    n=5,
    max_new_tokens=512,
)
assert all(seq.startswith("MKTLLLTLVVVTIVCLD") for seq in result["sequences"])
```

## Scoring Sequences

Evaluate how likely a sequence is under the model. ProGen3 scores bidirectionally (N→C and C→N averaged) for a more robust estimate:

```python
sequences = [
    "MKTLLLTLVVVTIVCLDLGYAAQSEGSSRQLIAAIGAICGAILLNYTFNQEIAQ",
    "ACDEFGHIKLMNPQRSTVWY",
]
scores = model.score(sequences)
print("Log-likelihoods:", scores["log_likelihood"])
print("Perplexities:", scores["perplexity"])
```

## How It Works

ProGen3 is a **causal language model** — it factorizes the probability of a protein as:

$$P(\text{seq}) = \prod_{i=1}^{L} P(a_i \mid a_1, \ldots, a_{i-1})$$

Each token is sampled from the model's predicted distribution given all previous tokens. The model uses a sparse mixture-of-experts (MoE) architecture.

When used with `sample()`, the `AutoregressiveLogitFormatter` adapts this to the framework's position-aligned convention by shifting logits left by one (so `logits[i]` = prediction for position `i`) and blocking all positions except the first unfilled one.

**Source**: [`examples/autoregressive_generation.py`](https://github.com/ishan-gaur/proteingen/blob/main/examples/autoregressive_generation.py)
