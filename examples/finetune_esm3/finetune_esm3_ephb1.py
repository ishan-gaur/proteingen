"""Fine-tune ESM3 with LoRA on the EphB1 kinase domain MSA.

Loads ~14k homologous sequences from a UniRef MSA, strips alignment gaps,
and continues training ESM3 with LoRA adapters using masked language modeling.

Optionally conditions on the 7KPM crystal structure. Structure conditioning
requires all sequences to be exactly 295 residues (the construct length),
so it filters the MSA accordingly. Without structure, all sequences >= min_length
are used.

The dataset API handles tokenization, padding, and noising via a collator
that is built from the model's tokenizer and noise functions.

Usage:
    uv run python examples/finetune_esm3/finetune_esm3_ephb1.py --device cuda
    uv run python examples/finetune_esm3/finetune_esm3_ephb1.py --device cuda --structure

Submitting to SLURM:
    bash ~/slurm/run_python.sh --uv examples/finetune_esm3/finetune_esm3_ephb1.py --device cuda
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader

from proteingen.data import (
    ProteinDataset,
    read_fasta,
    aligned_sequences_to_raw,
    uniform_mask_noise,
    uniform_time,
)
from proteingen.models.esm import ESM3

DATA_DIR = Path(__file__).parent
CONSTRUCT_LENGTH = 295  # EphB1 kinase domain construct (UniProt 602-896)


def load_msa_sequences(
    min_length: int = 200,
    exact_length: int | None = None,
) -> list[str]:
    """Load EphB1 MSA sequences, strip gaps, filter by length.

    Args:
        min_length: Minimum unaligned sequence length to include.
        exact_length: If set, keep only sequences of this exact length
            (used for structure conditioning which requires fixed length).
    """
    entries = read_fasta(str(DATA_DIR / "EphB1_MSA.fasta"))
    aligned = [seq for _, seq in entries]
    raw = aligned_sequences_to_raw(aligned)

    original_count = len(raw)
    if exact_length is not None:
        raw = [s for s in raw if len(s) == exact_length]
        print(
            f"Loaded {original_count} MSA sequences, "
            f"kept {len(raw)} with length == {exact_length}"
        )
    else:
        raw = [s for s in raw if len(s) >= min_length]
        print(
            f"Loaded {original_count} MSA sequences, "
            f"kept {len(raw)} with length >= {min_length}"
        )

    if raw:
        lengths = [len(s) for s in raw]
        print(
            f"  Length range: {min(lengths)}-{max(lengths)}, "
            f"mean: {sum(lengths) / len(lengths):.0f}"
        )
    return raw


def main():
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="Fine-tune ESM3 on EphB1 MSA")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--checkpoint", default="esm3-open", help="ESM3 checkpoint name"
    )
    parser.add_argument(
        "--structure", action="store_true", help="Condition on 7KPM structure"
    )
    parser.add_argument("--lora-r", type=int, default=4, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=8, help="LoRA alpha")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--min-length",
        type=int,
        default=200,
        help="Minimum sequence length to include",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Cap number of sequences (for quick testing)",
    )
    parser.add_argument(
        "--save-dir", type=str, default=None, help="Directory to save checkpoint"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="proteingen-finetune",
        help="Wandb project name",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use bfloat16 automatic mixed precision (recommended for GPU)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Data ──────────────────────────────────────────────────────────────
    if args.structure:
        sequences = load_msa_sequences(exact_length=CONSTRUCT_LENGTH)
        assert len(sequences) > 0, (
            f"No sequences of length {CONSTRUCT_LENGTH} in MSA. "
            "Structure conditioning requires exact-length sequences."
        )
    else:
        sequences = load_msa_sequences(min_length=args.min_length)
    if args.max_sequences is not None:
        sequences = sequences[: args.max_sequences]
        print(f"  Capped to {len(sequences)} sequences")

    dataset = ProteinDataset(sequences=sequences)

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"\nLoading {args.checkpoint}...")
    model = ESM3(args.checkpoint)

    if args.structure:
        print("Conditioning on 7KPM structure...")
        coords = torch.load(DATA_DIR / "7KPM_atom37_295.pt", weights_only=True)
        model.set_condition_({"coords_RAX": coords})

    model.apply_lora(r=args.lora_r, lora_alpha=args.lora_alpha)
    model.to(device)

    # Re-freeze any params that got unfrozen during structure encoding
    # (ESM3's VQ-VAE is loaded lazily by set_condition_)
    for name, p in model.named_parameters():
        if "lora_" not in name:
            p.requires_grad = False
        else:
            p.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"Parameters: {n_trainable:,} trainable / {n_total:,} total "
        f"({100 * n_trainable / n_total:.1f}%)"
    )

    # ── DataLoader ────────────────────────────────────────────────────────
    noise_fn = uniform_mask_noise(model.tokenizer)
    collate_fn = dataset.collator(
        model,
        noise_fn=noise_fn,
        time_sampler=uniform_time,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ── Wandb ─────────────────────────────────────────────────────────────
    run_name = f"esm3_ephb1_r{args.lora_r}"
    if args.structure:
        run_name += "_struct"

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "checkpoint": args.checkpoint,
            "structure": args.structure,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "min_length": args.min_length,
            "max_sequences": args.max_sequences,
            "n_sequences": len(sequences),
            "n_trainable_params": n_trainable,
            "n_total_params": n_total,
            "amp": args.amp,
        },
    )

    # ── Training ──────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    V = model.OUTPUT_DIM
    pad_id = model.tokenizer.pad_token_id

    use_amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else None

    print(
        f"\nTraining: {args.epochs} epochs, {len(loader)} batches/epoch, "
        f"batch_size={args.batch_size}, lr={args.lr}"
    )
    print(f"Structure conditioning: {args.structure}")
    print(f"AMP: {use_amp}\n")

    model.train()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)

            with torch.amp.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                # Forward with structure conditioning if set
                if model.observations is not None:
                    obs = model.collate_observations(input_ids, model.observations)
                    raw = model(input_ids, **obs)
                    logits = model.format_raw_to_logits(raw, input_ids, **obs)
                else:
                    raw = model(input_ids)
                    logits = model.format_raw_to_logits(raw, input_ids)

                loss = F.cross_entropy(
                    logits.float().reshape(-1, V),
                    target_ids.reshape(-1),
                    ignore_index=pad_id,
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

            n_tokens = (target_ids != pad_id).sum().item()
            batch_loss = loss.item()
            if not (torch.isnan(loss) or torch.isinf(loss)):
                epoch_loss += batch_loss * n_tokens
                epoch_tokens += n_tokens
            global_step += 1

            wandb.log(
                {
                    "train/loss": batch_loss,
                    "train/ppl": torch.exp(loss.detach().cpu()).item(),
                    "train/tokens": n_tokens,
                    "global_step": global_step,
                },
                step=global_step,
            )

            if (batch_idx + 1) % 100 == 0:
                avg = epoch_loss / max(epoch_tokens, 1)
                print(
                    f"  Epoch {epoch} | batch {batch_idx + 1}/{len(loader)} | "
                    f"loss: {avg:.4f} | ppl: {torch.exp(torch.tensor(avg)):.2f}",
                    flush=True,
                )

        avg_loss = epoch_loss / max(epoch_tokens, 1)
        ppl = torch.exp(torch.tensor(avg_loss)).item()
        elapsed = time.time() - t0

        wandb.log(
            {
                "epoch/loss": avg_loss,
                "epoch/ppl": ppl,
                "epoch/time_s": elapsed,
                "epoch": epoch,
            },
            step=global_step,
        )

        print(
            f"Epoch {epoch}/{args.epochs} | loss: {avg_loss:.4f} | "
            f"ppl: {ppl:.2f} | time: {elapsed:.1f}s",
            flush=True,
        )

    # ── Save ──────────────────────────────────────────────────────────────
    save_dir = args.save_dir or str(
        DATA_DIR / "checkpoints" / f"esm3_ephb1_lora_r{args.lora_r}"
    )
    print(f"\nSaving to {save_dir}")
    model.save(save_dir)

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
