# Contributing

ProteinGen is designed to make model contributions straightforward — especially when using an AI coding agent.

## Adding a new model

The recommended workflow is to prompt Claude Code (or another agent) with the appropriate skill file. These skill files explain step-by-step how to migrate a model into the ProteinGen framework, what tests to write, and what docs to update.

<!-- TODO[pi]: create these skill files in a `contributing/` folder in the repo and link them from AGENTS.md -->

### Adding a Generative Model

!!! note "Coming soon"
    Skill file: `contributing/add_generative_model.md`

A generative model contribution involves:

1. **Wrap the model** as a `TransitionModel` or `TransitionModelWithEmbedding` subclass
2. **Implement required methods**: `format_raw_to_logits`, and if using embeddings, `differentiable_embedding` + `embedding_to_outputs`
3. **Handle tokenization**: create or reuse a tokenizer, set up `MaskedModelLogitFormatter` if it's a masked LM
4. **Implement checkpointing**: `_save_args()`, and test `save()` / `from_checkpoint()` round-trip
5. **Write tests**: construction, forward pass shape, log-prob pipeline, gradient flow, batching, conditioning (if applicable), temperature scaling
6. **Add to docs**: entry in [Models](models.md) table, update `docs/reference/models.md` for API reference

### Adding a Predictive Model

!!! note "Coming soon"
    Skill file: `contributing/add_predictive_model.md`

A predictive model contribution involves:

1. **Subclass `PredictiveModel`** and implement `forward` + `format_raw_to_logits` (using the binary logit functions)
2. **Handle conditioning** if the model needs structure or other inputs: override `preprocess_observations` and `collate_observations`
3. **Handle tokenization**: use or create an appropriate tokenizer; if it differs from the generative model's tokenizer, TAG/DEG will handle translation via `GuidanceProjection`
4. **Implement checkpointing**: `_save_args()`, test round-trip
5. **Write tests**: construction, forward pass, `get_log_probs`, `grad_log_prob`, conditioning, checkpointing
6. **Add to docs**: entry in [Models](models.md) table

### Example: how we added ESMC

ESMC was migrated from the ESM library as follows:

- Subclassed `TransitionModelWithEmbedding`
- `differentiable_embedding`: OHE → `embed.weight` matmul → transformer → deep embeddings
- `embedding_to_outputs`: embeddings → `sequence_head`
- `MaskedModelLogitFormatter` handles masked-token logit masking
- Tests cover: construction, embed path vs forward match, gradient flow, log probs, batching, temperature, LoRA

## Code standards

- Follow the patterns in existing models — look at `src/proteingen/models/esm.py` as the reference implementation
- Use type hints on all function signatures
- Keep docstrings concise — don't restate what the code already says
- Run `uv run ruff check` and `uv run ruff format` before submitting
- Run `uv run python -m pytest tests/ -v` to verify nothing is broken

## Getting help

Open an issue on [GitHub](https://github.com/ishan-gaur/proteingen/issues) or prompt your agent with:

> "Read the ProteinGen AGENTS.md and the contributing skill file for [generative/predictive] models, then add [model name] following the documented pattern."
