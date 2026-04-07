"""Unconditional autoregressive protein generation with ProGen3.

Generates novel protein sequences using ProGen3's left-to-right sampling,
both via the standard sample() interface and via open-ended generate().

Requires: pip install git+https://github.com/Profluent-AI/progen3.git
GPU: any modern GPU with bfloat16 support (112M model fits on <8GB)
"""

from proteingen.models import ProGen3
from proteingen import sample

model = ProGen3("Profluent-Bio/progen3-112m").cuda()

# Fixed-length generation via sample() — 5 proteins of 100 residues each
print("=== Fixed-Length Generation (sample, left-to-right) ===")
init_x = ["<mask>" * 100 for _ in range(5)]
result = sample(model, init_x, in_order="left_to_right")
for i, seq in enumerate(result["sequences"]):
    print(f"Protein {i}: {len(seq)} residues")
    print(f"  {seq[:80]}...")
print()

# Open-ended generation — variable length
print("=== Open-Ended Generation (generate) ===")
result = model.generate(n=5, max_new_tokens=256, temperature=0.8, top_p=0.95)
for i, seq in enumerate(result["sequences"]):
    print(f"Protein {i}: {len(seq)} residues")
    print(f"  {seq[:80]}...")
print()

# Prompted generation — complete an N-terminal prefix
print("=== Prompted Generation (prefix: MKTLLLTLVVVTIVCLD) ===")
result = model.generate(prompt="MKTLLLTLVVVTIVCLD", n=3, max_new_tokens=512)
for i, seq in enumerate(result["sequences"]):
    print(f"Protein {i}: {len(seq)} residues")
    print(f"  {seq[:80]}...")
print()

# Score some sequences
print("=== Sequence Scoring ===")
sequences = [
    "MKTLLLTLVVVTIVCLDLGYAAQSEGSSRQLIAAIGAICGAILLNYTFNQEIAQ",
    "ACDEFGHIKLMNPQRSTVWY",
]
scores = model.score(sequences)
for seq, ll, ppl in zip(sequences, scores["log_likelihood"], scores["perplexity"]):
    print(f"  LL={ll:.3f}  PPL={ppl:.1f}  {seq[:40]}...")
