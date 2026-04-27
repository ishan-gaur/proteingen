# models

Concrete model implementations that subclass the core abstractions. Each model wraps an external library (ESM, ProteinMPNN) into ProteinGen's unified interface.

## Tokenization landscape

Three tokenizer ecosystems coexist in the library — cross-tokenizer mapping is handled by `GuidanceProjection` (see [guide](guide.md)):

| Tokenizer | Vocab size | Special tokens | Used by |
|-----------|-----------|----------------|---------|
| ESM (`EsmSequenceTokenizer`) | 33 | `<cls>=0, <pad>=1, <eos>=2, <unk>=3, <mask>=32` | ESMC, ESM3 |
| MPNN (`MPNNTokenizer`) | 21 (or 22 with mask) | `UNK(X)=20`, optional `<mask>=21` | StabilityPMPNN |
| Simple 20-AA | 21 | `pad=20` | Custom predictors |

---

## API Reference

::: proteingen.models
