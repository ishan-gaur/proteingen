# Autoregressive Generation (ProGen3)

Generate novel protein sequences using ProGen3's autoregressive (left-to-right) sampling. Unlike masked models that iteratively unmask positions, ProGen3 generates one amino acid at a time from N-terminal to C-terminal.

## Quick Start

```python
from proteingen.models import ProGen3

model = ProGen3("Profluent-Bio/progen3-112m").cuda()
result = model.generate(n=5, max_new_tokens=256)
for seq in result["sequences"]:
    print(f"Length {len(seq)}: {seq[:60]}...")
```

```bash
uv run python examples/autoregressive_generation.py
```

## Unconditional Generation

Start from nothing — the model decides both the sequence and its length:

```python
from proteingen.models import ProGen3

model = ProGen3("Profluent-Bio/progen3-112m").cuda()

# Generate 10 novel proteins, up to 512 residues each
result = model.generate(
    n=10,
    max_new_tokens=512,
    temperature=0.8,   # higher = more diverse
    top_p=0.95,        # nucleus sampling threshold
)

for i, seq in enumerate(result["sequences"]):
    print(f"Protein {i}: {len(seq)} residues")
    print(f"  {seq[:80]}...")
```

The model generates until it produces a natural stop signal (the C-terminal direction token followed by EOS). Each sequence will have a different length — the model learns protein length distributions from its training data.

## Prompted Generation (Sequence Completion)

Provide an N-terminal prefix and let the model complete the protein:

```python
# Complete a signal peptide into a full protein
result = model.generate(
    prompt="MKTLLLTLVVVTIVCLD",
    n=5,
    max_new_tokens=512,
)
# Each generated sequence starts with the prompt
assert all(seq.startswith("MKTLLLTLVVVTIVCLD") for seq in result["sequences"])
```

## Scoring Sequences

Evaluate how likely a sequence is under the model. ProGen3 scores bidirectionally (N→C and C→N averaged) for a more robust estimate:

```python
sequences = [
    "MKTLLLTLVVVTIVCLDLGYAAQSEGSSRQLIAAIGAICGAILLNYTFNQEIAQ",
    "ACDEFGHIKLMNPQRSTVWY",  # not a real protein
]
scores = model.score(sequences)
print("Log-likelihoods:", scores["log_likelihood"])
print("Perplexities:", scores["perplexity"])
```

## How It Works

ProGen3 is a **causal language model** — it factorizes the probability of a protein as:

$$P(\text{seq}) = \prod_{i=1}^{L} P(a_i \mid a_1, \ldots, a_{i-1})$$

Each token is sampled from the model's predicted distribution given all previous tokens. This is the same generation strategy used in GPT-style text models, applied to protein sequences.

The model uses a sparse mixture-of-experts (MoE) architecture where each layer routes tokens to a subset of expert networks, allowing the model to scale to billions of parameters while keeping inference efficient.

### Encoding format

Internally, sequences are framed with direction tokens:

```
<bos> 1 M K T L ... 2 <eos>
```

Where `1` = N→C direction and `2` = C→N direction. The wrapper handles this framing automatically.

**Source**: [`examples/autoregressive_generation.py`](https://github.com/ishan-gaur/proteingen/blob/main/examples/autoregressive_generation.py)
