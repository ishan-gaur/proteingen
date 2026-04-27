"""Initialize an EmbeddingMLP with PCA-projected ESMC embeddings.

Demonstrates how to use ``EmbeddingMLP.init_embed_from_pretrained_pca``
to transfer learned amino acid representations from a pretrained masked
language model (ESMC-300m) into a small MLP suitable for downstream tasks.

ESMC's 960-dimensional token embeddings encode useful amino acid
similarities, but are far too large for a lightweight predictor. The method
compresses them via PCA and handles the vocabulary mapping between the two
models automatically.
"""

import torch
from types import SimpleNamespace
from torch.nn import functional as F
from esm.models.esmc import ESMC
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from proteingen.modeling import EmbeddingMLP, binary_logits

# ── 1. Define the target vocabulary (20 standard amino acids + padding) ──────

STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"
target_vocab = {aa: i for i, aa in enumerate(STANDARD_AAS)}
VOCAB_SIZE = len(target_vocab) + 1  # 21: indices 0–19 for AAs, 20 for padding
PADDING_IDX = 20
N_COMPONENTS = 20

# ── 2. Build the model ──────────────────────────────────────────────────────

tokenizer = SimpleNamespace(vocab_size=VOCAB_SIZE, pad_token_id=PADDING_IDX)


class DemoEmbeddingMLP(EmbeddingMLP):
    """Concrete subclass for demonstration — uses binary logits."""

    def format_raw_to_logits(self, raw_output, ohe_seq_SPT, **kwargs):
        return binary_logits(raw_output.reshape(-1))


model = DemoEmbeddingMLP(
    tokenizer=tokenizer,
    sequence_length=50,
    embed_dim=N_COMPONENTS,
    model_dim=128,
    n_layers=2,
    output_dim=1,
)

# ── 3. Initialize embeddings from ESMC PCA ──────────────────────────────────

esmc = ESMC.from_pretrained("esmc_300m", device=torch.device("cpu"))
esm_tokenizer = EsmSequenceTokenizer()

model.init_embed_from_pretrained_pca(
    source=esmc.embed,
    source_vocab=esm_tokenizer.vocab,
    target_vocab=target_vocab,
)

print(f"Embedding initialized from ESMC PCA ({N_COMPONENTS} components):")
print(f"  Shape:       {model.embed.weight.shape}")
print(f"  Padding row: all zeros = {(model.embed.weight[PADDING_IDX] == 0).all().item()}")
print(f"  Learnable:   {model.embed.weight.requires_grad}")

# ── 4. Verify with a dummy forward pass ─────────────────────────────────────

dummy_seqs = torch.randint(0, len(STANDARD_AAS), (4, 50))
ohe = F.one_hot(dummy_seqs, num_classes=VOCAB_SIZE).float()
output = model(ohe)
print(f"\nForward pass:  input {dummy_seqs.shape} → OHE {ohe.shape} → output {output.shape}")
