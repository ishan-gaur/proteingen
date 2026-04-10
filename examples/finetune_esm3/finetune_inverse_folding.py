"""Fine-tune ESM3 as an inverse folding model on EphB1 domain structures.

Takes pre-computed AF3 structures (from fold_msa_domains.py) and trains ESM3
with LoRA to predict masked sequence positions given the full structure.
Each sequence has its own structure of matching length.

Evaluates structure-conditioned likelihood curves on a held-out eval set
at the end of each epoch.

Usage:
    uv run python examples/finetune_esm3/finetune_inverse_folding.py --device cuda
    uv run python examples/finetune_esm3/finetune_inverse_folding.py --device cuda --structures ephb1_structures.pt

Submitting to SLURM:
    bash ~/slurm/run_python.sh --uv "$(pwd)/examples/finetune_esm3/finetune_inverse_folding.py" --device cuda --amp
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, Dataset
from esm.utils.constants import esm3 as ESM3_CONSTANTS
from protstar.eval.likelihood_curves import (
    LogProbTrajectory,
    plot_log_prob_trajectories,
)
from protstar.models.esm import ESM3

DATA_DIR = Path(__file__).parent


# ── Dataset ───────────────────────────────────────────────────────────────


class InverseFoldingDataset(Dataset):
    """Dataset of (sequence, structure) pairs for inverse folding training.

    Each sample has its own structure tokens and coordinates, pre-computed
    by fold_msa_domains.py. Lengths may vary across samples.
    """

    def __init__(
        self,
        sequences: list[str],
        structure_tokens: list[torch.Tensor],  # each (L+2,)
        coordinates: list[torch.Tensor],  # each (L+2, 37, 3)
    ):
        assert len(sequences) == len(structure_tokens) == len(coordinates)
        for i, (seq, st, co) in enumerate(
            zip(sequences, structure_tokens, coordinates)
        ):
            assert st.shape[0] == len(seq) + 2, (
                f"Sample {i}: structure_tokens length {st.shape[0]} != seq length {len(seq)} + 2"
            )
            assert co.shape[0] == len(seq) + 2

        self.sequences = sequences
        self.structure_tokens = structure_tokens
        self.coordinates = coordinates

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "sequence": self.sequences[idx],
            "structure_tokens": self.structure_tokens[idx],
            "coordinates": self.coordinates[idx],
        }


def inverse_folding_collator(tokenizer, mask_token_id: int):
    """Collator that tokenizes sequences, masks them, and pads structures.

    For inverse folding: mask ALL non-special sequence positions. The model
    must predict the entire sequence from structure alone.
    """
    special_ids = set(tokenizer.all_special_ids)
    structure_pad = ESM3_CONSTANTS.STRUCTURE_PAD_TOKEN

    def collate_fn(batch: list[dict]) -> dict:
        sequences = [s["sequence"] for s in batch]

        # Tokenize sequences with padding
        tokenized = tokenizer(sequences, padding=True, return_tensors="pt")
        target_ids = tokenized["input_ids"]  # (B, L) — the true tokens
        B, L = target_ids.shape

        # Mask ALL non-special positions (full inverse folding)
        input_ids = target_ids.clone()
        maskable = torch.ones(B, L, dtype=torch.bool)
        for sid in special_ids:
            maskable &= target_ids != sid
        input_ids[maskable] = mask_token_id

        # Pad structure tokens and coordinates to match tokenized length L
        # Tokenizer adds BOS and EOS, structures already have BOS/EOS from VQ-VAE
        padded_struct_tokens = torch.full((B, L), structure_pad, dtype=torch.long)
        coord_shape = batch[0]["coordinates"].shape[-2:]  # (37, 3)
        padded_coords = torch.zeros(B, L, *coord_shape)

        for i, sample in enumerate(batch):
            st = sample["structure_tokens"]
            co = sample["coordinates"]
            seq_len = st.shape[0]  # L_i + 2 (with BOS/EOS)
            assert seq_len <= L, f"Structure length {seq_len} > padded length {L}"
            padded_struct_tokens[i, :seq_len] = st
            padded_coords[i, :seq_len] = co

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "structure_tokens": padded_struct_tokens,
            "coordinates": padded_coords,
            "maskable": maskable,
        }

    return collate_fn


# ── Likelihood curve evaluation ───────────────────────────────────────────


@torch.no_grad()
def compute_eval_log_probs(
    model: ESM3,
    dataset: InverseFoldingDataset,
    indices: list[int],
    n_time_points: int,
    batch_size: int,
    device: torch.device,
    use_structure: bool = True,
) -> LogProbTrajectory:
    """Compute log-prob trajectories, optionally with structure conditioning.

    Args:
        use_structure: if True, conditions on per-sequence structures.
            If False, runs sequence-only (no structure tokens/coords).
    """
    tokenizer = model.tokenizer
    mask_token_id = tokenizer.mask_token_id
    special_ids = set(tokenizer.all_special_ids)

    sequences = [dataset.sequences[i] for i in indices]

    tokenized = tokenizer(sequences, padding=True, return_tensors="pt")
    true_tokens = tokenized["input_ids"]  # (S, P)
    S, P = true_tokens.shape

    # Prepare structures if needed
    padded_struct = None
    padded_coords = None
    if use_structure:
        struct_tokens_list = [dataset.structure_tokens[i] for i in indices]
        coords_list = [dataset.coordinates[i] for i in indices]
        structure_pad = ESM3_CONSTANTS.STRUCTURE_PAD_TOKEN
        padded_struct = torch.full((S, P), structure_pad, dtype=torch.long)
        padded_coords = torch.zeros(S, P, 37, 3)
        for i in range(S):
            sl = struct_tokens_list[i].shape[0]
            padded_struct[i, :sl] = struct_tokens_list[i]
            padded_coords[i, :sl] = coords_list[i]

    # Maskable positions
    maskable = torch.ones(S, P, dtype=torch.bool)
    for sid in special_ids:
        maskable &= true_tokens != sid

    time_points = torch.linspace(0, 1, n_time_points + 1)[:-1]
    avg_log_probs = torch.full((S, n_time_points), float("nan"))

    model.eval()
    for t_idx, t in enumerate(time_points):
        keep = torch.rand(S, P) < t
        to_mask = maskable & ~keep

        noised = true_tokens.clone()
        noised[to_mask] = mask_token_id

        log_prob_chunks = []
        for start in range(0, S, batch_size):
            end = min(start + batch_size, S)
            inp = noised[start:end].to(device)

            if use_structure:
                st = padded_struct[start:end].to(device)
                co = padded_coords[start:end].to(device)
                raw = model(inp, structure_tokens=st, coordinates=co)
                logits = model.format_raw_to_logits(
                    raw, inp, structure_tokens=st, coordinates=co
                )
            else:
                raw = model(inp)
                logits = model.format_raw_to_logits(raw, inp)

            lp = F.log_softmax(logits.float(), dim=-1)
            log_prob_chunks.append(lp.cpu())

        log_probs = torch.cat(log_prob_chunks, dim=0)  # (S, P, V)
        V = log_probs.shape[-1]
        safe_idx = true_tokens.clamp(max=V - 1)
        true_lp = log_probs.gather(2, safe_idx.unsqueeze(2)).squeeze(2)

        true_lp[~to_mask] = 0.0
        n_masked_per_seq = to_mask.sum(dim=1).float()
        seq_avg = true_lp.sum(dim=1) / n_masked_per_seq.clamp(min=1)
        seq_avg[n_masked_per_seq == 0] = float("nan")
        avg_log_probs[:, t_idx] = seq_avg

    model.train()
    return LogProbTrajectory(time_points=time_points, avg_log_probs=avg_log_probs)


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        description="Fine-tune ESM3 as inverse folding model on EphB1 structures"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--structures",
        default=str(DATA_DIR / "ephb1_structures.pt"),
        help="Path to pre-computed structures from fold_msa_domains.py",
    )
    parser.add_argument(
        "--checkpoint", default="esm3-open", help="ESM3 checkpoint name"
    )
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use bfloat16 automatic mixed precision",
    )
    parser.add_argument(
        "--n-eval", type=int, default=30, help="Number of sequences for eval"
    )
    parser.add_argument(
        "--n-time-points",
        type=int,
        default=15,
        help="Time points for likelihood curves",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="protstar-finetune",
    )
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Cap training sequences (for testing)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Load pre-computed structures ──────────────────────────────────────
    print(f"Loading structures from {args.structures}...")
    data = torch.load(args.structures, weights_only=False)
    sequences = data["sequences"]
    structure_tokens = data["structure_tokens"]
    coordinates = data["coordinates"]
    ranking_scores = data["ranking_scores"]
    print(f"  {len(sequences)} sequences with structures")
    print(
        f"  AF3 ranking scores: min={min(ranking_scores):.3f}, "
        f"max={max(ranking_scores):.3f}, mean={sum(ranking_scores) / len(ranking_scores):.3f}"
    )

    if args.max_sequences is not None:
        sequences = sequences[: args.max_sequences]
        structure_tokens = structure_tokens[: args.max_sequences]
        coordinates = coordinates[: args.max_sequences]
        print(f"  Capped to {len(sequences)} sequences")

    # Split: last n_eval for evaluation, rest for training
    n_eval = min(args.n_eval, len(sequences) // 5)  # at most 20% for eval
    train_seqs = sequences[:-n_eval] if n_eval > 0 else sequences
    train_st = structure_tokens[:-n_eval] if n_eval > 0 else structure_tokens
    train_co = coordinates[:-n_eval] if n_eval > 0 else coordinates

    eval_dataset = None
    eval_indices = None
    if n_eval > 0:
        eval_seqs = sequences[-n_eval:]
        eval_st = structure_tokens[-n_eval:]
        eval_co = coordinates[-n_eval:]
        eval_dataset = InverseFoldingDataset(eval_seqs, eval_st, eval_co)
        eval_indices = list(range(len(eval_dataset)))
        print(f"  Train: {len(train_seqs)}, Eval: {len(eval_seqs)}")

    train_dataset = InverseFoldingDataset(train_seqs, train_st, train_co)

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"\nLoading {args.checkpoint}...")
    model = ESM3(args.checkpoint)
    model.apply_lora(r=args.lora_r, lora_alpha=args.lora_alpha)
    model.to(device)

    for name, p in model.named_parameters():
        p.requires_grad = "lora_" in name

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"Parameters: {n_trainable:,} trainable / {n_total:,} total "
        f"({100 * n_trainable / n_total:.1f}%)"
    )

    # ── DataLoader ────────────────────────────────────────────────────────
    collate_fn = inverse_folding_collator(
        model.tokenizer, model.tokenizer.mask_token_id
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ── Wandb ─────────────────────────────────────────────────────────────
    run_name = f"esm3_ephb1_IF_r{args.lora_r}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "task": "inverse_folding",
            "checkpoint": args.checkpoint,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "amp": args.amp,
            "n_train": len(train_dataset),
            "n_eval": n_eval,
            "n_trainable_params": n_trainable,
        },
    )

    # ── Eval helper ─────────────────────────────────────────────────────
    def run_eval(epoch_label: str, step: int | None = None):
        """Compute structure-conditioned and sequence-only likelihood curves."""
        results = {}
        for mode, use_struct in [("struct", True), ("seq_only", False)]:
            traj = compute_eval_log_probs(
                model,
                eval_dataset,
                eval_indices,
                n_time_points=args.n_time_points,
                batch_size=args.batch_size,
                device=device,
                use_structure=use_struct,
            )
            results[mode] = traj
            mean_lp = torch.nanmean(traj["avg_log_probs"], dim=0)
            log_dict = {
                f"eval/{mode}/log_prob_t0": mean_lp[0].item(),
                f"eval/{mode}/log_prob_mid": mean_lp[len(mean_lp) // 2].item(),
            }
            if step is not None:
                wandb.log(log_dict, step=step)
            else:
                wandb.log(log_dict)
            print(
                f"  {epoch_label} [{mode}]: log p at t=0: {mean_lp[0]:.3f}, "
                f"t=0.5: {mean_lp[len(mean_lp) // 2]:.3f}"
            )
        return results

    # ── Initial eval ──────────────────────────────────────────────────────
    if eval_dataset is not None:
        print("\nComputing initial likelihood curves...")
        init_results = run_eval("Init")

    # ── Training ──────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    use_amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else None

    print(
        f"\nTraining: {args.epochs} epochs, {len(loader)} batches/epoch, "
        f"batch_size={args.batch_size}, lr={args.lr}, AMP={use_amp}\n"
    )

    model.train()
    global_step = 0
    struct_trajectories = []
    struct_labels = []
    seq_only_trajectories = []
    seq_only_labels = []

    if eval_dataset is not None:
        struct_trajectories.append(init_results["struct"])
        struct_labels.append("epoch 0 (pretrained) + struct")
        seq_only_trajectories.append(init_results["seq_only"])
        seq_only_labels.append("epoch 0 (pretrained) seq-only")

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            struct_tokens = batch["structure_tokens"].to(device)
            coords = batch["coordinates"].to(device)
            maskable = batch["maskable"]

            with torch.amp.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                raw = model(
                    input_ids,
                    structure_tokens=struct_tokens,
                    coordinates=coords,
                )
                logits = model.format_raw_to_logits(
                    raw,
                    input_ids,
                    structure_tokens=struct_tokens,
                    coordinates=coords,
                )

                # Loss on all maskable (= all non-special) positions
                loss = F.cross_entropy(
                    logits.float()[maskable],
                    target_ids[maskable],
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

            n_masked = maskable.sum().item()
            batch_loss = loss.item()
            if not (torch.isnan(loss) or torch.isinf(loss)):
                epoch_loss += batch_loss * n_masked
                epoch_tokens += n_masked
            global_step += 1

            wandb.log(
                {
                    "train/loss": batch_loss,
                    "train/ppl": torch.exp(loss.detach().cpu()).item(),
                    "train/n_masked": n_masked,
                },
                step=global_step,
            )

            if (batch_idx + 1) % 50 == 0:
                avg = epoch_loss / max(epoch_tokens, 1)
                print(
                    f"  Epoch {epoch} | batch {batch_idx + 1}/{len(loader)} | "
                    f"loss: {avg:.4f} | ppl: {torch.exp(torch.tensor(avg)):.2f}"
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
            f"ppl: {ppl:.2f} | time: {elapsed:.1f}s"
        )

        # ── End-of-epoch likelihood curves ────────────────────────────────
        if eval_dataset is not None:
            print(f"  Computing likelihood curves (epoch {epoch})...")
            epoch_results = run_eval(f"Epoch {epoch}", step=global_step)
            struct_trajectories.append(epoch_results["struct"])
            struct_labels.append(f"epoch {epoch} + struct")
            seq_only_trajectories.append(epoch_results["seq_only"])
            seq_only_labels.append(f"epoch {epoch} seq-only")

    # ── Save model ────────────────────────────────────────────────────────
    save_dir = args.save_dir or str(
        DATA_DIR / "checkpoints" / f"esm3_ephb1_IF_lora_r{args.lora_r}"
    )
    print(f"\nSaving to {save_dir}")
    model.save(save_dir)

    # ── Save likelihood curve plots ──────────────────────────────────────
    if struct_trajectories:
        # Combined plot: structure-conditioned vs sequence-only, first & last epoch
        combined_trajs = [
            seq_only_trajectories[0],
            struct_trajectories[0],
            seq_only_trajectories[-1],
            struct_trajectories[-1],
        ]
        combined_labels = [
            seq_only_labels[0],
            struct_labels[0],
            seq_only_labels[-1],
            struct_labels[-1],
        ]
        plot_path = str(DATA_DIR / "outputs" / "inverse_folding_likelihood_curves.png")
        plot_log_prob_trajectories(
            trajectories=combined_trajs,
            labels=combined_labels,
            output_path=plot_path,
            show_individual=True,
            title="ESM3 inverse folding: struct-conditioned vs seq-only",
        )
        wandb.log({"eval/likelihood_curves": wandb.Image(plot_path)})
        print(f"Saved likelihood curve plot to {plot_path}")

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
