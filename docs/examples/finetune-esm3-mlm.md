# Fine-tuning ESM3 on EphB1 (Sequence-only MLM)

??? abstract "Architecture Breakdown"
    **Data:** ~10k EphB1 kinase domain homologs from a UniRef MSA, loaded as a `ProteinDataset` → [data](../reference/data.md). Uses `uniform_mask_noise` + `uniform_time` for training-time masking. MSA was obtained via jackhmmer → [MSA Acquisition](../workflows/msa-acquisition.md), then cleaned → [MSA → Dataset](../workflows/msa-to-dataset.md).

    **Models:** ESM3 with LoRA adapters (r=4) → [generative_modeling](../reference/generative_modeling.md). This is an instance of the [Fine-tuning](../workflows/finetune-generative.md) module.

    **Sampling:** None (training only).

    **Evaluation:** Training loss/perplexity tracked per epoch. Likelihood curves would be the natural next step → [Likelihood Curves](../workflows/likelihood-curves.md).

Fine-tune ESM3 with LoRA on ~10k EphB1 kinase domain homologs using masked language modeling.

## Quick Start

```bash
uv run python examples/finetune_esm3/finetune_esm3_ephb1.py --device cuda --amp --epochs 5
```

## What It Does

This trains the model to predict randomly masked amino acids from surrounding sequence context. The training data is an MSA of EphB1 kinase domain homologs (~10k sequences, 200–295 residues).

**Results after 5 epochs**: loss 1.80 → 1.60, perplexity 6.04 → 4.96.

## Key Details

- Uses **LoRA** adapters (not full fine-tuning) to keep memory usage manageable
- Requires `--amp` flag (bfloat16) — ESM3 fp32 logits overflow without it
- Training uses `model(input_ids, **observations)` directly, not `get_log_probs`

See the [Fine-tuning workflow](../workflows/finetune-generative.md) for details on the training loop and LoRA setup.

**Source**: [`examples/finetune_esm3/finetune_esm3_ephb1.py`](https://github.com/ishan-gaur/protstar/blob/main/examples/finetune_esm3/finetune_esm3_ephb1.py)
