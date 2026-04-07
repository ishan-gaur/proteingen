"""Step 4: Analyze benchmark results — metrics, plots, and comparisons.

Computes:
    - Sequence recovery (% identity to original)
    - Per-step log-probability trajectories during generation
    - AF3 pLDDT and pTM scores
    - TM-score to original structure (requires AF3 CIF files)
    - Fold class classification from secondary structure content (DSSP on AF3 outputs)
    - Scaling analysis within model families

Produces:
    - Likelihood trajectory plots per masking level
    - pLDDT vs masking level (bar charts per model)
    - TM-score vs masking level
    - Sequence recovery vs masking level
    - Fold class agreement analysis
    - Model scaling analysis plots

Usage:
    uv run python examples/benchmark_model_families/analyze.py
"""

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_DIR,
    FOLD_CLASSES,
    MASK_FRACTIONS,
    MODEL_CONFIGS,
    OUTPUT_DIR,
)

# Colors for model families
FAMILY_COLORS = {
    "esmc": "#1f77b4",
    "esm3": "#2ca02c",
    "dplm2": "#d62728",
}

# Distinct colors per model
MODEL_COLORS = {
    "ESMC-300M": "#1f77b4",
    "ESMC-600M": "#aec7e8",
    "ESM3-Open": "#2ca02c",
    "DPLM2-150M": "#d62728",
    "DPLM2-650M": "#ff9896",
    "DPLM2-3B": "#e377c2",
}

PLOT_DIR = OUTPUT_DIR / "plots"


# ── Utilities ────────────────────────────────────────────────────────────


def sequence_identity(seq1: str, seq2: str) -> float:
    """Fraction of positions with identical amino acids."""
    min_len = min(len(seq1), len(seq2))
    if min_len == 0:
        return 0.0
    matches = sum(a == b for a, b in zip(seq1[:min_len], seq2[:min_len]))
    return matches / max(len(seq1), len(seq2))


def classify_fold(helix_frac: float, strand_frac: float) -> str:
    """Classify fold based on secondary structure composition."""
    for name, predicate in FOLD_CLASSES.items():
        if predicate(helix_frac, strand_frac):
            return name
    return "small/other"


def parse_dssp_from_cif(cif_path: Path) -> dict:
    """Extract secondary structure fractions from an AF3 mmCIF file using biotite.

    Returns {"helix_frac": float, "strand_frac": float, "coil_frac": float}.
    """
    try:
        import biotite.structure.io.pdbx as pdbx

        pdbx_file = pdbx.CIFFile.read(str(cif_path))
        structure = pdbx.get_structure(pdbx_file, model=1)

        # Use annotated secondary structure if available, else estimate from backbone
        if "sec_struct_type" in structure._annot:
            sse = structure._annot["sec_struct_type"]
        else:
            # Estimate from CA positions using biotite's DSSP-like annotation
            ca_mask = structure.atom_name == "CA"
            n_res = int(ca_mask.sum())
            if n_res == 0:
                return {"helix_frac": 0.0, "strand_frac": 0.0, "coil_frac": 1.0}
            # Without full DSSP, use a simple estimate from coordinates
            return {"helix_frac": 0.0, "strand_frac": 0.0, "coil_frac": 1.0}

        n_res = len(sse)
        helix = sum(1 for s in sse if s in ("H", "G", "I"))
        strand = sum(1 for s in sse if s in ("E", "B"))
        coil = n_res - helix - strand
        return {
            "helix_frac": helix / max(n_res, 1),
            "strand_frac": strand / max(n_res, 1),
            "coil_frac": coil / max(n_res, 1),
        }
    except Exception as e:
        warnings.warn(f"DSSP parsing failed for {cif_path}: {e}")
        return {"helix_frac": 0.0, "strand_frac": 0.0, "coil_frac": 1.0}


def extract_plddt_from_cif(cif_path: Path) -> float:
    """Extract global pLDDT from an AF3 mmCIF file.

    AF3 stores pLDDT in the _ma_qa_metric_global category.
    """
    try:
        with open(cif_path) as f:
            for line in f:
                if line.startswith("_ma_qa_metric_global.metric_value"):
                    # Value is on the same line after the key
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        return float(parts[1])
                    # Or on the next line
                    break
            # Also try parsing as a loop
            for line in f:
                line = line.strip()
                if line and not line.startswith("_") and not line.startswith("#"):
                    try:
                        return float(line.split()[0])
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("_") or line.startswith("#"):
                    break
        return float("nan")
    except Exception as e:
        warnings.warn(f"pLDDT extraction failed for {cif_path}: {e}")
        return float("nan")


def compute_tm_score(cif_path1: Path, cif_path2: Path) -> float:
    """Compute TM-score between two structures using tmtools."""
    try:
        import biotite.structure.io.pdbx as pdbx
        import tmtools

        def get_ca_coords_and_seq(path):
            pdbx_file = pdbx.CIFFile.read(str(path))
            structure = pdbx.get_structure(pdbx_file, model=1)
            ca_mask = structure.atom_name == "CA"
            ca = structure[ca_mask]
            coords = ca.coord.astype(np.float64)
            # Build sequence from residue names
            from biotite.structure import get_residues

            _, res_names = get_residues(ca)
            three_to_one = {
                "ALA": "A",
                "CYS": "C",
                "ASP": "D",
                "GLU": "E",
                "PHE": "F",
                "GLY": "G",
                "HIS": "H",
                "ILE": "I",
                "LYS": "K",
                "LEU": "L",
                "MET": "M",
                "ASN": "N",
                "PRO": "P",
                "GLN": "Q",
                "ARG": "R",
                "SER": "S",
                "THR": "T",
                "VAL": "V",
                "TRP": "W",
                "TYR": "Y",
            }
            seq = "".join(three_to_one.get(r, "X") for r in res_names)
            return coords, seq

        coords1, seq1 = get_ca_coords_and_seq(cif_path1)
        coords2, seq2 = get_ca_coords_and_seq(cif_path2)

        result = tmtools.tm_align(coords1, coords2, seq1, seq2)
        return float(result.tm_norm_chain1)

    except Exception as e:
        warnings.warn(f"TM-score failed: {e}")
        return float("nan")


# ── Data loading ─────────────────────────────────────────────────────────


def load_all_results() -> dict:
    """Load all generation results, fold results, and original sequences."""
    data = {}

    # Original sequences
    with open(DATA_DIR / "sequences.json") as f:
        data["entries"] = json.load(f)
    data["sequences"] = [e["sequence"] for e in data["entries"]]

    # Generation results per model
    data["generation"] = {}
    for model_dir in sorted(OUTPUT_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        results_path = model_dir / "generation_results.json"
        if not results_path.exists():
            continue
        with open(results_path) as f:
            data["generation"][model_dir.name] = json.load(f)

    # Fold results
    fold_path = OUTPUT_DIR / "fold_results" / "fold_results.json"
    if fold_path.exists():
        with open(fold_path) as f:
            data["fold"] = json.load(f)
    else:
        data["fold"] = {}

    return data


def build_metrics_table(data: dict) -> list[dict]:
    """Build a flat table of per-sample metrics for analysis."""
    rows = []
    fold_results = data.get("fold", {})
    cif_dir = OUTPUT_DIR / "fold_results" / "cif_files"

    for model_name, gen_results in data["generation"].items():
        # Find model config
        config = None
        for family, display, ckpt, params in MODEL_CONFIGS:
            if display.replace(" ", "_") == model_name:
                config = (family, display, ckpt, params)
                break
        if config is None:
            continue

        family, display_name, _, n_params = config

        for key, result in gen_results.items():
            seq_idx = result["seq_idx"]
            order_idx = result["order_idx"]
            mask_frac = result["mask_fraction"]
            gen_seq = result["generated_sequence"]
            orig_seq = data["sequences"][seq_idx]

            # Sequence identity
            seq_id = sequence_identity(gen_seq, orig_seq)

            # Step log probs
            step_lps = [
                lp
                for lp in result["step_log_probs"]
                if lp is not None and not np.isnan(lp)
            ]
            mean_step_lp = np.mean(step_lps) if step_lps else float("nan")

            row = {
                "model": display_name,
                "family": family,
                "n_params": n_params,
                "seq_idx": seq_idx,
                "order_idx": order_idx,
                "mask_fraction": mask_frac,
                "accession": data["entries"][seq_idx]["accession"],
                "original_length": len(orig_seq),
                "generated_sequence": gen_seq,
                "sequence_identity": seq_id,
                "mean_step_log_prob": mean_step_lp,
                "step_log_probs": step_lps,
            }

            # Add fold metrics if available
            fold_key = f"{model_name}_{key}"
            if fold_key in fold_results and "error" not in fold_results[fold_key]:
                fr = fold_results[fold_key]
                confidences = fr.get("summary_confidences", {})

                row["ptm"] = confidences.get("ptm", float("nan"))
                row["ranking_score"] = fr.get("best_ranking_score", float("nan"))

                # pLDDT from CIF file (B-factor field)
                gen_cif = cif_dir / f"{fold_key}.cif"
                if gen_cif.exists():
                    row["plddt"] = extract_plddt_from_cif(gen_cif)
                else:
                    row["plddt"] = float("nan")

                # TM-score to original (if both CIF files exist)
                orig_cif = (
                    cif_dir
                    / f"original_{seq_idx}_{data['entries'][seq_idx]['accession']}.cif"
                )
                if gen_cif.exists() and orig_cif.exists():
                    row["tm_score"] = compute_tm_score(gen_cif, orig_cif)
                else:
                    row["tm_score"] = float("nan")

                # Fold classification
                if gen_cif.exists():
                    dssp = parse_dssp_from_cif(gen_cif)
                    row["fold_class"] = classify_fold(
                        dssp["helix_frac"], dssp["strand_frac"]
                    )
                    row["helix_frac"] = dssp["helix_frac"]
                    row["strand_frac"] = dssp["strand_frac"]
            else:
                row["plddt"] = float("nan")
                row["ptm"] = float("nan")
                row["ranking_score"] = float("nan")
                row["tm_score"] = float("nan")

            rows.append(row)

    return rows


# ── Plotting ─────────────────────────────────────────────────────────────


def plot_likelihood_trajectories(metrics: list[dict]):
    """Plot per-step log-probability curves grouped by mask fraction and model.

    2×2 grid, mean curve + shaded ±1 std region only (no individual trajectories).
    """
    n = len(MASK_FRACTIONS)
    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10), sharey=True)
    axes = axes.flatten()

    for idx, mf in enumerate(MASK_FRACTIONS):
        ax = axes[idx]
        for model_name, color in MODEL_COLORS.items():
            model_rows = [
                r
                for r in metrics
                if r["model"] == model_name and r["mask_fraction"] == mf
            ]
            if not model_rows:
                continue

            all_lps = [
                row["step_log_probs"] for row in model_rows if row["step_log_probs"]
            ]
            if not all_lps:
                continue

            common_grid = np.linspace(0, 1, 50)
            interp_lps = []
            for lps in all_lps:
                xs = np.linspace(0, 1, len(lps))
                interp_lps.append(np.interp(common_grid, xs, lps))
            mean_lp = np.mean(interp_lps, axis=0)
            std_lp = np.std(interp_lps, axis=0)
            ax.plot(common_grid, mean_lp, color=color, linewidth=2, label=model_name)
            ax.fill_between(
                common_grid,
                mean_lp - std_lp,
                mean_lp + std_lp,
                alpha=0.2,
                color=color,
            )

        ax.set_title(f"Mask = {mf:.0%}", fontsize=12)
        ax.set_xlabel("Fraction of masked positions filled")
        if idx % ncols == 0:
            ax.set_ylabel("Log p(sampled token)")

    # Hide unused subplots if MASK_FRACTIONS < 4
    for idx in range(n, nrows * ncols):
        axes[idx].set_visible(False)

    # Single legend outside the grid
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="center right", bbox_to_anchor=(1.12, 0.5), fontsize=9
    )
    fig.suptitle("Generation log-likelihood trajectories", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "likelihood_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {PLOT_DIR / 'likelihood_trajectories.png'}")


def plot_metric_vs_masking(
    metrics: list[dict], metric_key: str, ylabel: str, filename: str
):
    """Boxplot of a metric across models and masking levels."""
    models = list(MODEL_COLORS.keys())
    available_models = [m for m in models if any(r["model"] == m for r in metrics)]

    n_models = len(available_models)
    n_fracs = len(MASK_FRACTIONS)
    fig, ax = plt.subplots(figsize=(max(12, n_fracs * 3), 5))

    # Group width and box width
    group_width = 0.8
    box_width = group_width / n_models
    positions_all = []
    colors_all = []
    data_all = []
    tick_positions = []

    for j, mf in enumerate(MASK_FRACTIONS):
        group_center = j
        tick_positions.append(group_center)
        for i, model_name in enumerate(available_models):
            vals = [
                r[metric_key]
                for r in metrics
                if r["model"] == model_name
                and r["mask_fraction"] == mf
                and not np.isnan(r.get(metric_key, float("nan")))
            ]
            if not vals:
                vals = [float("nan")]
            offset = (i - n_models / 2 + 0.5) * box_width
            positions_all.append(group_center + offset)
            colors_all.append(MODEL_COLORS[model_name])
            data_all.append(vals)

    bp = ax.boxplot(
        data_all,
        positions=positions_all,
        widths=box_width * 0.85,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"linewidth": 1},
        capprops={"linewidth": 1},
    )

    for patch, color in zip(bp["boxes"], colors_all):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    # Legend: one entry per model
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=MODEL_COLORS[m], alpha=0.8)
        for m in available_models
    ]
    ax.legend(legend_handles, available_models, fontsize=8)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{mf:.0%}" for mf in MASK_FRACTIONS])
    ax.set_xlabel("Mask fraction")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs. masking level")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {PLOT_DIR / filename}")


def plot_scaling_analysis(metrics: list[dict]):
    """How model scaling affects generation quality at different masking levels.

    Uses boxplots. All families shown (including single-model families like ESM3).
    """
    families = defaultdict(list)
    for family, display_name, _, n_params in MODEL_CONFIGS:
        families[family].append((display_name, n_params))

    if not families:
        print("No families found, skipping scaling analysis")
        return

    for metric_key, ylabel in [
        ("sequence_identity", "Sequence Identity"),
        ("mean_step_log_prob", "Mean Step Log Prob"),
        ("plddt", "AF3 pLDDT"),
        ("tm_score", "TM-score"),
    ]:
        fig, axes = plt.subplots(
            1, len(MASK_FRACTIONS), figsize=(4 * len(MASK_FRACTIONS), 4), sharey=True
        )
        if len(MASK_FRACTIONS) == 1:
            axes = [axes]

        for ax, mf in zip(axes, MASK_FRACTIONS):
            for family, models in families.items():
                sorted_models = sorted(models, key=lambda x: x[1])
                param_counts = []
                box_data = []

                for model_name, n_params in sorted_models:
                    vals = [
                        r[metric_key]
                        for r in metrics
                        if r["model"] == model_name
                        and r["mask_fraction"] == mf
                        and not np.isnan(r.get(metric_key, float("nan")))
                    ]
                    if vals:
                        param_counts.append(n_params)
                        box_data.append(vals)

                if not param_counts:
                    continue

                color = FAMILY_COLORS[family]
                # Box width in log-space: fraction of the axis range
                log_positions = [np.log10(p) for p in param_counts]
                log_width = 0.08  # width in log10 units

                bp = ax.boxplot(
                    box_data,
                    positions=log_positions,
                    widths=log_width,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={"color": "black", "linewidth": 1.5},
                    whiskerprops={"color": color, "linewidth": 1},
                    capprops={"color": color, "linewidth": 1},
                    manage_ticks=False,
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                # Connect medians with a line for families with >1 model
                if len(param_counts) > 1:
                    medians = [np.median(d) for d in box_data]
                    ax.plot(
                        log_positions,
                        medians,
                        color=color,
                        linewidth=1.5,
                        alpha=0.6,
                        zorder=0,
                    )

            # Format x-axis as log with parameter labels
            all_params = sorted({p for ms in families.values() for _, p in ms})
            ax.set_xticks([np.log10(p) for p in all_params])
            ax.set_xticklabels([f"{p}M" for p in all_params], fontsize=7, rotation=45)
            ax.set_xlabel("Parameters")
            ax.set_title(f"Mask = {mf:.0%}")
            if ax == axes[0]:
                ax.set_ylabel(ylabel)

        # Legend: one entry per family
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=FAMILY_COLORS[f], alpha=0.7)
            for f in families
        ]
        legend_labels = [f.upper() for f in families]
        axes[-1].legend(legend_handles, legend_labels, fontsize=8)

        fig.suptitle(f"Scaling: {ylabel}", fontsize=13, y=1.02)
        fig.tight_layout()
        fname = f"scaling_{metric_key}.png"
        fig.savefig(PLOT_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {PLOT_DIR / fname}")


def plot_fold_class_analysis(metrics: list[dict], data: dict):
    """Analyze fold class agreement between generated and original proteins."""
    # Get original fold classes from CIF files
    cif_dir = OUTPUT_DIR / "fold_results" / "cif_files"

    original_classes = {}
    for i, entry in enumerate(data["entries"]):
        orig_key = f"original_{i}_{entry['accession']}"
        orig_cif = cif_dir / f"{orig_key}.cif"
        if orig_cif.exists():
            dssp = parse_dssp_from_cif(orig_cif)
            original_classes[i] = classify_fold(dssp["helix_frac"], dssp["strand_frac"])

    if not original_classes:
        print("No original fold classes available, skipping fold class analysis")
        return

    # Compute fold class agreement per model and masking level
    models = list(MODEL_COLORS.keys())
    available_models = [m for m in models if any(r["model"] == m for r in metrics)]

    fig, ax = plt.subplots(figsize=(12, 5))
    n_models = len(available_models)
    bar_width = 0.8 / n_models
    x = np.arange(len(MASK_FRACTIONS))

    for i, model_name in enumerate(available_models):
        agreements = []
        for mf in MASK_FRACTIONS:
            model_rows = [
                r
                for r in metrics
                if r["model"] == model_name
                and r["mask_fraction"] == mf
                and "fold_class" in r
                and r["seq_idx"] in original_classes
            ]
            if model_rows:
                agree = sum(
                    1
                    for r in model_rows
                    if r["fold_class"] == original_classes[r["seq_idx"]]
                )
                agreements.append(agree / len(model_rows))
            else:
                agreements.append(0)

        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            agreements,
            bar_width,
            label=model_name,
            color=MODEL_COLORS[model_name],
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{mf:.0%}" for mf in MASK_FRACTIONS])
    ax.set_xlabel("Mask fraction")
    ax.set_ylabel("Fold class agreement")
    ax.set_title("Fold class agreement with original protein")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "fold_class_agreement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {PLOT_DIR / 'fold_class_agreement.png'}")


def print_summary_table(metrics: list[dict]):
    """Print a text summary table of results."""
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)

    header = f"{'Model':<15} {'Mask%':>6} {'SeqID':>7} {'MeanLP':>8} {'pLDDT':>7} {'pTM':>6} {'TM-sc':>7} {'N':>4}"
    print(header)
    print("-" * len(header))

    models = list(MODEL_COLORS.keys())
    for model_name in models:
        for mf in MASK_FRACTIONS:
            rows = [
                r
                for r in metrics
                if r["model"] == model_name and r["mask_fraction"] == mf
            ]
            if not rows:
                continue

            def safe_mean(key):
                vals = [r[key] for r in rows if not np.isnan(r.get(key, float("nan")))]
                return np.mean(vals) if vals else float("nan")

            seq_id = safe_mean("sequence_identity")
            mean_lp = safe_mean("mean_step_log_prob")
            plddt = safe_mean("plddt")
            ptm = safe_mean("ptm")
            tm = safe_mean("tm_score")

            print(
                f"{model_name:<15} {mf:>5.0%} {seq_id:>7.3f} {mean_lp:>8.3f} "
                f"{plddt:>7.2f} {ptm:>6.3f} {tm:>7.3f} {len(rows):>4d}"
            )
        print()


def main():
    sys.stdout.reconfigure(line_buffering=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    data = load_all_results()
    print(f"  {len(data['entries'])} original sequences")
    print(f"  {len(data['generation'])} models with generation results")
    print(f"  {len(data['fold'])} fold results")

    if not data["generation"]:
        print("\nNo generation results found. Run generate.py first.")
        return

    print("\nBuilding metrics table...")
    metrics = build_metrics_table(data)
    print(f"  {len(metrics)} total samples")

    # Save metrics table as JSON
    metrics_path = OUTPUT_DIR / "metrics.json"
    # Strip step_log_probs for the saved version (too large)
    metrics_save = [
        {k: v for k, v in r.items() if k != "step_log_probs"} for r in metrics
    ]
    with open(metrics_path, "w") as f:
        json.dump(metrics_save, f, indent=2)
    print(f"  Saved metrics to {metrics_path}")

    # Print summary
    print_summary_table(metrics)

    # Generate plots
    print("\nGenerating plots...")
    plot_likelihood_trajectories(metrics)
    plot_metric_vs_masking(
        metrics, "sequence_identity", "Sequence Identity", "sequence_identity.png"
    )
    plot_metric_vs_masking(
        metrics, "mean_step_log_prob", "Mean Step Log Prob", "mean_step_log_prob.png"
    )

    if any(not np.isnan(r.get("plddt", float("nan"))) for r in metrics):
        plot_metric_vs_masking(metrics, "plddt", "AF3 pLDDT", "plddt.png")
        plot_metric_vs_masking(metrics, "ptm", "AF3 pTM", "ptm.png")
        plot_metric_vs_masking(
            metrics, "tm_score", "TM-score to Original", "tm_score.png"
        )
        plot_fold_class_analysis(metrics, data)

    plot_scaling_analysis(metrics)

    print(f"\nAll plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
