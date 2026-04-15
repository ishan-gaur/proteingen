# Frame2seq

Frame2seq is a structure-conditioned inverse folding model ([Akpinaroglu et al.](https://doi.org/10.1101/2023.12.15.571823)).

- **Class**: `protstar.models.Frame2seq`
- **Source**: [dakpinaroglu/Frame2seq](https://github.com/dakpinaroglu/Frame2seq)
- **Conditioning**: required (`pdb_path`, `chain_id`)
- **Output dim**: 22 (20 canonical AAs + blocked `X` + blocked `<mask>`)
- **Embedding dim**: `128 * n_checkpoints` (default ensemble: 384)

`Frame2seq` loads the bundled checkpoint ensemble from the `frame2seq` package and averages logits across members.

## Basic usage

```python
import torch
from protstar.models import Frame2seq

model = Frame2seq()
model.set_condition_({"pdb_path": "1YCR.pdb", "chain_id": "A"})

L = model.observations["X_LA3"].shape[0]
seq = torch.full((1, L), model.tokenizer.mask_token_id, dtype=torch.long)

log_probs = model.get_log_probs(seq)  # (1, L, 22)
```

## Conditioning schema

```python
class Frame2seqConditioning(TypedDict):
    pdb_path: str | Path
    chain_id: str
```

- `pdb_path`: path to a PDB file
- `chain_id`: single chain identifier to design (e.g. `"A"`)

You can also build this dict via:

```python
conditioning = Frame2seq.condition_from_pdb("1YCR.pdb", "A")
```

## Notes

- Conditioning is required; calling `get_log_probs` without `set_condition_` raises `ValueError`.
- Masked inputs are supported: the wrapper's `<mask>` input token is mapped internally to Frame2seq's native `X` (unknown/masked) input channel.
- Output logits for `X` and `<mask>` are blocked (`-inf`), so sampling/log-probs normalize over the 20 canonical amino acids.
- Frame2seq checkpoints are loaded via PyTorch Lightning and may print a one-time checkpoint-upgrade notice.
