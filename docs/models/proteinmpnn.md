# ProteinMPNN

Structure-conditioned autoregressive sequence design model ([Dauparas et al., 2022](https://www.science.org/doi/10.1126/science.add2187)). Wraps the Foundry implementation (`rc-foundry[all]`) as a `GenerativeModelWithEmbedding`.

- **Output dim**: 22 (20 standard AAs + UNK + mask — UNK and mask columns are always -inf)
- **Embedding dim**: 128
- **Parameters**: 1.7M (small, runs fast on CPU)
- **Structure conditioning**: **required** — the model is a structure-conditioned inverse folding model
- **LoRA support**: yes, via `apply_lora()`

## Available checkpoints

From Foundry registry:

| Checkpoint | Description |
|---|---|
| `proteinmpnn` | Standard ProteinMPNN (default) |
| `solublempnn` | Trained on soluble proteins only |

```python
from protstar.models import ProteinMPNN

model = ProteinMPNN("proteinmpnn")  # or "solublempnn"
```

## Structure conditioning

ProteinMPNN **requires** backbone structure as input. Pass a `PDBStructure` from `load_pdb`:

```python
from protstar.models.utils import load_pdb

structure = load_pdb("1YCR.pdb")

# Set conditioning — runs graph featurization + encoder once
model.set_condition_({"structure": structure})

# Get log probabilities for a sequence
tokens = model.tokenizer("A" * 98)["input_ids"]
log_probs = model.get_log_probs(tokens)  # (1, 98, 22)

# Or use context manager
with model.conditioned_on({"structure": structure}):
    log_probs = model.get_log_probs(tokens)
```

For multi-chain structures, use `design_chains` to specify which chains to design (others are held as fixed context):

```python
# Design only chain B, use chain A as structural context
model.set_condition_({"structure": structure, "design_chains": ["B"]})
```

## How the wrapper works

ProteinMPNN natively decodes one residue at a time in a random autoregressive order. The wrapper instead runs the decoder in **conditional-minus-self** mode: each position's prediction is conditioned on every other position's sequence identity and the full structure, but not on its own identity. This produces a pseudo-likelihood P(residue_i | structure, all other residues) at every position simultaneously, making the model behave like a masked language model compatible with the library's `get_log_probs` / sampling / TAG interface. Importantly, this means `get_log_probs` returns a real conditional distribution at *every* position — not just mask positions — so the output is directly useful for scoring sequences.

The MPNN architecture has a natural split: the encoder processes backbone geometry (no sequence information) and the decoder predicts sequence conditioned on the encoder output. Since structure doesn't change between calls, `set_condition_()` runs graph featurization and the encoder once, caching node features, edge features, and graph topology. Every subsequent call only runs the lightweight 3-layer decoder. MPNN natively outputs 21-dim logits (20 AAs + UNK); the wrapper pads to 22-dim with a -inf mask column for compatibility with the tokenizer, and the logit formatter sets UNK to -inf so only the 20 standard amino acids have finite probability.

<!-- TODO: compare pseudo-likelihood (conditional-minus-self) vs any-order autoregressive decoding on ~10 random PDBs from the MPNN test set -->

!!! info "Validated against Foundry"
    The wrapper is tested against Foundry's own MPNN pipeline on PDB 1YCR (p53/MDM2, 2 chains, 98 residues) and produces **bitwise-identical logits** — 0.0 max absolute difference, 100% argmax agreement across all positions.
