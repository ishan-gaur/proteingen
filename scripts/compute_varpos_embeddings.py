"""Compute per-variable-position ESMC embeddings for TrpB dataset."""
import torch
import numpy as np
from pathlib import Path
import time
import sys; sys.path.insert(0, ".")

from proteingen.models.esm import ESMC
from examples.trpb_linear_probe import load_trpb_data, find_variable_positions

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="esmc_600m")
parser.add_argument("--batch-size", type=int, default=32)
args = parser.parse_args()

df = load_trpb_data()
variable_pos = find_variable_positions(df)
var_pos_tokens = [p + 1 for p in variable_pos]  # +1 for CLS
print(f"Variable positions: {variable_pos}, token positions: {var_pos_tokens}", flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}, Checkpoint: {args.checkpoint}", flush=True)
esmc = ESMC(args.checkpoint)
esmc.to(device)
tokenizer = esmc.tokenizer
cache_dir = Path(f"data/trpb_embeddings/{args.checkpoint}_varpos")
cache_dir.mkdir(parents=True, exist_ok=True)
print(f"Model loaded, EMB_DIM={esmc.EMB_DIM}", flush=True)

for split in ["train", "valid", "test"]:
    cache_path = cache_dir / f"{split}.pt"
    if cache_path.exists():
        print(f"{split}: already cached at {cache_path}", flush=True)
        continue

    split_seqs = df[df["stage"] == split]["protein"].tolist()
    all_embs = []
    t0 = time.time()
    for start in range(0, len(split_seqs), args.batch_size):
        batch = split_seqs[start : start + args.batch_size]
        tokens = tokenizer(batch, padding=True, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            emb = esmc.embed(tokens)
        var_emb = emb[:, var_pos_tokens, :].reshape(len(batch), -1).cpu()
        all_embs.append(var_emb)
        done = min(start + args.batch_size, len(split_seqs))
        if (start // args.batch_size) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {split}: {done:>6d} / {len(split_seqs)} ({elapsed:.0f}s)", flush=True)

    result = torch.cat(all_embs)
    torch.save(result, cache_path)
    print(f"  {split}: saved {result.shape} in {time.time()-t0:.0f}s", flush=True)

print("Done", flush=True)
