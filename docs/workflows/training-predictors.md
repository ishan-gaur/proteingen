# Training Predictors

Train oracle and noisy predictive models from assay-labeled data for use in guided generation.

## Two models, two roles

Guided generation with ProteinGuide requires two predictive models:

| Model | Training data | Input type | Role |
|-------|--------------|------------|------|
| **Oracle** | All available data (all rounds) | Clean sequences | Evaluation only — scores final generated sequences |
| **Noisy predictor** | Current round data | Randomly masked sequences | Used during sampling — must handle partial sequences |

The oracle answers "how good is this fully-designed sequence?". The noisy predictor answers "given what I can see so far, is this sequence likely to be good?" — which is what the sampler needs at each unmasking step.

## Training the oracle

The oracle is trained on clean (fully unmasked) sequences with standard supervised learning. It should be as accurate as possible — use all available data including later experimental rounds.

```python
oracle = MyPredictor(tokenizer=gen_model.tokenizer, ...)
# Standard training loop: MSE or cross-entropy on clean sequences
for batch in oracle_loader:
    pred = oracle.forward(batch["ohe"])
    loss = F.mse_loss(pred, batch["labels"])
    loss.backward()
    optimizer.step()
```

**Architecture choice:** Use whatever works best on your data. OneHotMLP is a strong default for small datasets (< 5k sequences). LinearProbe on frozen ESMC embeddings works well for larger datasets.

## Training the noisy predictor

The noisy predictor is trained identically to the oracle *except* that input sequences are randomly masked at each training step. This makes it robust to the partially-masked sequences it sees during guided generation.

```python
noisy_predictor = MyPredictor(tokenizer=gen_model.tokenizer, ...)
for batch in train_loader:
    tokens = batch["tokens"].clone()
    
    # Random masking: mask fraction ~ Uniform(0, 1) each step
    t = torch.rand(1).item()
    mask = torch.rand(tokens.shape) < t
    mask[:, 0] = False   # preserve BOS
    mask[:, -1] = False  # preserve EOS
    tokens[mask] = mask_token_id
    
    ohe = F.one_hot(tokens, vocab_size).float()
    pred = noisy_predictor.forward(ohe)
    loss = F.mse_loss(pred, batch["labels"])
    loss.backward()
    optimizer.step()
```

**Key detail:** The masking distribution during training should match the sampling schedule. If you use `uniform_mask_noise` for generation, train with uniform `t`. Validate on clean sequences (no masking) — this is the regime at the end of generation where the predictor's accuracy matters most.

## Validating predictor–oracle agreement

Before using the noisy predictor for guidance, check that it agrees with the oracle on clean sequences:

```python
from scipy.stats import spearmanr

oracle_scores = oracle.predict(val_sequences)
predictor_scores = noisy_predictor.predict(val_sequences)
rho, _ = spearmanr(oracle_scores, predictor_scores)
print(f"Agreement: ρ = {rho:.3f}")
```

**If agreement is low (ρ < 0.5):** The predictor can't be trusted during generation. Consider:

- More training data
- Simpler architecture (less overfitting)
- Different masking schedule
- Collecting more experimental data before attempting guided generation

See the [PbrR walkthrough](../examples/pbrr-walkthrough.md) for a complete implementation with oracle–predictor agreement plots.

## Choosing `format_raw_to_logits`

The binary logit function determines how raw predictions become guidance signals:

| Function | Best for | TAG-compatible? |
|----------|----------|-----------------|
| `point_estimate_binary_logits` | Simple thresholding | Only with small `k` (5–10) |
| `gaussian_binary_logits` | Uncertainty-aware predictions | Yes — differentiable through mean and variance |
| `binary_logits` | Direct classification | Yes |
| `categorical_binary_logits` | Multi-class (one vs rest) | Yes |

If using TAG guidance, prefer `gaussian_binary_logits` — it provides smooth gradients. For DEG guidance, `point_estimate_binary_logits` works fine regardless of `k` since DEG only needs rankings.

<!-- TODO: add training scripts as reusable functions in proteingen.train -->
