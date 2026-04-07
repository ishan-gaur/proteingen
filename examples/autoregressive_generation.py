"""Unconditional autoregressive protein generation with ProGen3.

Generates novel protein sequences from scratch using ProGen3's left-to-right
autoregressive sampling. The model decides both sequence content and length.

Requires: pip install git+https://github.com/Profluent-AI/progen3.git
GPU: any modern GPU with bfloat16 support (112M model fits on <8GB)
"""

from proteingen.models import ProGen3

model = ProGen3("Profluent-Bio/progen3-112m").cuda()

# Unconditional generation — 5 novel proteins, up to 256 residues each
result = model.generate(n=5, max_new_tokens=256, temperature=0.8, top_p=0.95)

print("=== Unconditional Generation ===")
for i, seq in enumerate(result["sequences"]):
    print(f"Protein {i}: {len(seq)} residues")
    print(f"  {seq[:80]}...")
print()

# Prompted generation — complete an N-terminal prefix
result = model.generate(
    prompt="MKTLLLTLVVVTIVCLD",
    n=3,
    max_new_tokens=512,
)

print("=== Prompted Generation (prefix: MKTLLLTLVVVTIVCLD) ===")
for i, seq in enumerate(result["sequences"]):
    print(f"Protein {i}: {len(seq)} residues")
    print(f"  {seq[:80]}...")
print()

# Score some sequences
sequences = [
    "MKTLLLTLVVVTIVCLDLGYAAQSEGSSRQLIAAIGAICGAILLNYTFNQEIAQ",
    "ACDEFGHIKLMNPQRSTVWY",
]
scores = model.score(sequences)
print("=== Sequence Scoring ===")
for seq, ll, ppl in zip(sequences, scores["log_likelihood"], scores["perplexity"]):
    print(f"  LL={ll:.3f}  PPL={ppl:.1f}  {seq[:40]}...")
