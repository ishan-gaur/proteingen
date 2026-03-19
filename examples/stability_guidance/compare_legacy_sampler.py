"""Head-to-head sampler comparison on DFM ESM3 + stability predictor.

Compares:
1) DFM linear-interpolation sampler on TAG posterior logits
2) Legacy flow-matching rate sampler with original guidance-ratio update

This is intended to isolate integration-loop differences when guidance is strong.
"""

from pathlib import Path
import argparse

import numpy as np
import torch

from dfm.models.esm import ESM3IF
from dfm.models.rocklin_ddg.stability_predictor import (
    PreTrainedStabilityPredictor,
    StabilityPMPNN,
)
from dfm.models.utils import pdb_to_atom37_and_seq
from dfm.guide import TAG
from dfm.sampling import (
    sample_linear_interpolation,
    sample_flow_matching_legacy,
    build_legacy_predictor_log_prob,
)


def predict_stability_raw(oracle_model, sequences, cond, device):
    """Get raw stability predictions for ddG computation."""
    from dfm.generative_modeling import MPNNTokenizer

    tokenizer = MPNNTokenizer()
    tokens = tokenizer(sequences)["input_ids"].to(device)
    B = tokens.shape[0]

    X = cond["X"][:1].expand(B, -1, -1, -1).to(device)
    mask = cond["mask"][:1].expand(B, -1).to(device)
    chain_M = cond["chain_M"][:1].expand(B, -1).to(device)
    residue_idx = cond["residue_idx"][:1].expand(B, -1).to(device)
    chain_encoding_all = cond["chain_encoding_all"][:1].expand(B, -1).to(device)

    with torch.no_grad():
        pred = oracle_model(X, tokens, mask, chain_M, residue_idx, chain_encoding_all)
    return pred.reshape(-1).cpu().numpy()


def esm_if_constraints(logits_SPT: torch.FloatTensor, xt_SP: torch.LongTensor) -> torch.FloatTensor:
    """Original demo token constraints for ESM3 inverse folding."""
    logits = logits_SPT.clone()
    logits[:, 0, 0] = 0.0
    logits[:, 0, 1:] = -float("inf")
    logits[:, -1, 2] = 0.0
    logits[:, -1, :2] = -float("inf")
    logits[:, -1, 3:] = -float("inf")
    logits[:, 1:-1, 0:4] = -float("inf")
    logits[:, 1:-1, 24:] = -float("inf")
    return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--x1-temp", type=float, default=0.1)
    parser.add_argument("--guide-temp", type=float, default=0.01)
    parser.add_argument("--stochasticity", type=float, default=0.0)
    args = parser.parse_args()

    device = args.device
    seed = args.seed
    num_samples = args.num_samples
    batch_size = args.batch_size
    dt = args.dt
    x1_temp = args.x1_temp
    guide_temp = args.guide_temp
    stochasticity = args.stochasticity

    base_dir = Path(__file__).resolve().parent
    pdb_path = base_dir / "data" / "structures" / "5KPH.pdb"
    oracle_path = base_dir / "weights" / "stability_regression.pt"
    classifier_path = base_dir / "weights" / "noisy_classifier_146_iter_1.pt"

    print("Loading DFM models...")
    esm_model = ESM3IF().to(device)
    classifier = (
        PreTrainedStabilityPredictor(str(classifier_path), one_hot_encode_input=True)
        .to(device)
        .eval()
    )
    classifier.set_temp_(guide_temp)
    tag_model = TAG(esm_model, classifier, use_clean_classifier=False).to(device)

    oracle_model = StabilityPMPNN.init(num_letters=21, vocab=21)
    oracle_model.load_state_dict(torch.load(str(oracle_path), weights_only=False))
    oracle_model = oracle_model.to(device).eval()

    print("Preparing conditioning...")
    coords_RAX, wt_seq = pdb_to_atom37_and_seq(str(pdb_path), backbone_only=True)
    esm_model.set_condition_({"coords_RAX": coords_RAX})
    classifier_cond = PreTrainedStabilityPredictor.prepare_conditioning(
        str(pdb_path), device=device
    )
    classifier.set_condition_(classifier_cond)

    masked = ["<mask>" * len(wt_seq)] * num_samples

    print("Sampling with linear interpolation sampler...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    guided_linear = sample_linear_interpolation(
        tag_model,
        masked,
        n_steps=int(1 / dt),
        return_string=True,
    )

    print("Sampling with legacy flow-matching rate sampler...")
    predictor_log_prob = build_legacy_predictor_log_prob(tag_model)
    torch.manual_seed(seed)
    np.random.seed(seed)
    guided_legacy = []
    for i in range(0, num_samples, batch_size):
        batch_masked = masked[i : i + batch_size]
        batch_samples = sample_flow_matching_legacy(
            model=esm_model,
            x_SP=batch_masked,
            dt=dt,
            predictor_log_prob=predictor_log_prob,
            guide_temp=guide_temp,
            use_tag=True,
            x1_temp=x1_temp,
            stochasticity=stochasticity,
            logits_postprocess=esm_if_constraints,
            return_string=True,
        )
        guided_legacy.extend(batch_samples)

    print("Evaluating ddG...")
    linear_pred = predict_stability_raw(oracle_model, guided_linear, classifier_cond, device)
    legacy_pred = predict_stability_raw(oracle_model, guided_legacy, classifier_cond, device)
    wt_pred = predict_stability_raw(oracle_model, [str(wt_seq)], classifier_cond, device)[0]

    linear_ddg = -1.0 * (linear_pred - wt_pred)
    legacy_ddg = -1.0 * (legacy_pred - wt_pred)

    same_seq_frac = float(
        np.mean(np.array(guided_linear, dtype=object) == np.array(guided_legacy, dtype=object))
    )

    print("\n=== Sampler Comparison ===")
    print(f"num_samples={num_samples}, dt={dt}, guide_temp={guide_temp}, x1_temp={x1_temp}")
    print(
        f"linear_interp: mean_ddg={np.mean(linear_ddg):.3f}, median_ddg={np.median(linear_ddg):.3f}, frac_ddg<=0={(linear_ddg <= 0).mean():.3f}"
    )
    print(
        f"legacy_rates:  mean_ddg={np.mean(legacy_ddg):.3f}, median_ddg={np.median(legacy_ddg):.3f}, frac_ddg<=0={(legacy_ddg <= 0).mean():.3f}"
    )
    print(f"same_sequence_fraction={same_seq_frac:.3f}")
    print(f"mean_abs_ddg_diff={np.mean(np.abs(linear_ddg - legacy_ddg)):.3f}")


if __name__ == "__main__":
    main()
