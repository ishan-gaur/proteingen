# PbrR Guided Protein Design Walkthrough

End-to-end walkthrough extending the [PbrR discrete guidance experiment](https://github.com/protein-discrete-guidance/pdg/pbrr/) from the paper *"Active learning-guided optimization of cell-free biosensors for lead testing in drinking water"* (Nature Communications, 2025).

PbrR is a transcription factor whose metal selectivity is being engineered toward lead (Pb) and away from zinc (Zn). The original experiment used a kernel ridge regression predictor with Bayesian guidance on ESMC to generate PbrR variants. This walkthrough extends that work through three steps using the `proteingen` library.

## Data

- **Round 1 data**: `PbrR_Pb_Zn_FC.csv` — 1,099 variants with Pb and Zn fold-change measurements
- **All-rounds data**: Extracted from [Supplementary Data](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-025-66964-6/MediaObjects/41467_2025_66964_MOESM6_ESM.xlsx) — ~2,024 variants including later experimental rounds
- **Train/test split**: Uses the saved `splits_dict.pt` from the original experiment for exact reproducibility

## Step 1: LoRA Finetune ESMC on Pareto Front

Finetunes ESMC-300M using LoRA adapters on the 33 "successful" training variants (those on the Pareto front: high Pb sensitivity, low Zn response). Compares pretrained vs finetuned model likelihoods on all remaining sequences, visualized as Pb vs Zn scatter plots colored by likelihood.

```bash
uv run python examples/pbrr_walkthrough/step1_finetune_esmc.py --epochs 100 --device cuda
```

**Outputs:**
- `outputs/step1_likelihood_scatter.png` — Side-by-side Pb vs Zn colored by model likelihood
- `outputs/step1_likelihood_diff.png` — Likelihood difference (finetuned − pretrained)
- `checkpoints/step1_lora/lora_adapter/` — Saved LoRA weights

## Step 2: Train and Compare Predictive Models

Trains three predictive models on the same train/test split as the original paper:

1. **Linear Probe** — ESMC embeddings → linear head (2 outputs: Pb, Zn log-FC)
2. **OHE-encoded MLP** — One-hot encoded sequences → MLP
3. **XGBoost** — One-hot features → gradient-boosted trees (new model added to `proteingen`)

Evaluates each on the held-out test set using RMSE, R², and Spearman ρ.

```bash
uv run python examples/pbrr_walkthrough/step2_train_predictors.py --device cuda
```

**Outputs:**
- `outputs/step2_model_comparison.png` — Bar charts comparing all metrics
- `outputs/step2_scatter_plots.png` — Predicted vs actual scatter plots
- `checkpoints/step2_results.pt` — Saved metrics

## Step 3: Oracle, Noisy Classifier, and Guided Generation

1. Trains an **oracle** on ALL PbrR data (rounds 1 + later rounds from the Nature paper supplementary) using the best model from Step 2
2. Trains a **noisy classifier** on round 1 data with random masking, for use with DEG guidance
3. Generates sequences using four methods and scores them with the oracle:
   - **Pretrained unconditional ESMC** — base model, no guidance
   - **Finetuned unconditional ESMC** — LoRA-adapted from Step 1
   - **DEG-guided pretrained ESMC** — guidance from noisy classifier
   - **DEG-guided finetuned ESMC** — guidance + finetuning combined

Only the SSM positions (experimentally mutated positions) are masked during generation, matching the original experiment.

```bash
uv run python examples/pbrr_walkthrough/step3_guided_generation.py \
    --device cuda \
    --oracle-type xgboost \
    --n-sequences 100
```

**Outputs:**
- `outputs/step3_generation_comparison.png` — 2×2 panel, each method vs dataset
- `outputs/step3_combined_overlay.png` — All methods overlaid
- `outputs/step3_success_rates.png` — Bar chart of % sequences in target region
- `checkpoints/step3_generations.pt` — All generated sequences and oracle scores

## New Library Feature: XGBoost Predictive Model

This walkthrough adds `XGBoostPredictor` to `proteingen.models` — a wrapper that integrates XGBoost regressors/classifiers into the `PredictiveModel` framework. Key properties:

- Operates on one-hot encoded sequences (same interface as `OneHotMLP`)
- Compatible with **DEG** guidance (enumeration-based, no gradients needed)
- **Not** compatible with TAG guidance (XGBoost is non-differentiable)
- Multi-output support (trains separate models per output dimension)

```python
from proteingen.models import XGBoostPredictor

class MyPredictor(XGBoostPredictor):
    def __init__(self):
        super().__init__(tokenizer=my_tokenizer, output_dim=2)
    
    def format_raw_to_logits(self, raw_output, ohe_seq_SPT, **kwargs):
        return point_estimate_binary_logits(raw_output[:, 0], self.target)

predictor = MyPredictor()
predictor.fit(train_ohe, train_labels)
predictions = predictor.predict(token_ids)
```

## Directory Structure

```
pbrr_walkthrough/
├── README.md                      # This file
├── data_utils.py                  # Shared data loading/preprocessing
├── step1_finetune_esmc.py         # LoRA finetuning + likelihood plots
├── step2_train_predictors.py      # Train and compare 3 predictive models
├── step3_guided_generation.py     # Oracle + noisy classifier + guided generation
├── outputs/                       # Generated figures (not tracked)
└── checkpoints/                   # Model checkpoints (not tracked)
```
