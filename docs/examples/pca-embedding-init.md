# PCA Embedding Initialization

Initialize a small `EmbeddingMLP` predictor using PCA-compressed embeddings from ESMC. This transfers learned amino acid representations from the pretrained model into a lightweight head.

## Quick Start

```bash
uv run python examples/pca_embedding_init.py
```

## How It Works

ESMC's 960-dim token embeddings encode useful amino acid similarities, but are too large for a small predictor. `init_embed_from_pretrained_pca` compresses them via PCA and handles vocabulary mapping automatically.

The result is a compact embedding layer that starts from a meaningful representation rather than random initialization — useful when training data is limited.

**Source**: [`examples/pca_embedding_init.py`](https://github.com/ishan-gaur/proteingen/blob/main/examples/pca_embedding_init.py)
