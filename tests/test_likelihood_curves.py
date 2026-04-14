"""Tests for protstar.eval.likelihood_curves.

Tests compute_log_prob_trajectory and plot_log_prob_trajectories using
mock generative models (deterministic and uniform) so tests are fast,
reproducible, and don't require downloading real models.
"""

import torch
import pytest
from torch import nn

from protstar.generative_modeling import GenerativeModel, PassThroughLogitFormatter
from protstar.eval.likelihood_curves import (
    compute_decoding_log_prob_trajectory,
    compute_log_prob_trajectory,
    plot_decoding_log_prob_trajectories,
    plot_log_prob_trajectories,
)


# ---------------------------------------------------------------------------
# Mock tokenizer — same pattern as test_sampling.py
# ---------------------------------------------------------------------------


class MockTokenizer:
    """Tiny tokenizer: tokens 0-4 are amino acids, 5=pad, 6=cls, 7=eos, 8=mask."""

    VOCAB_SIZE = 9

    def __init__(self):
        self.pad_token_id = 5
        self.cls_token_id = 6
        self.eos_token_id = 7
        self.mask_token_id = 8
        self.vocab = {str(i): i for i in range(5)}
        self.vocab.update({"<pad>": 5, "<cls>": 6, "<eos>": 7, "<mask>": 8})
        self.added_tokens_decoder = {5: "<pad>", 6: "<cls>", 7: "<eos>", 8: "<mask>"}

    @property
    def vocab_size(self):
        return self.VOCAB_SIZE

    def __call__(self, sequences, padding=True, return_tensors="pt"):
        encoded = []
        for seq in sequences:
            ids = [self.cls_token_id]
            ids.extend(int(c) for c in seq)
            ids.append(self.eos_token_id)
            encoded.append(ids)
        max_len = max(len(e) for e in encoded)
        for e in encoded:
            while len(e) < max_len:
                e.append(self.pad_token_id)
        return {"input_ids": torch.tensor(encoded, dtype=torch.long)}


# ---------------------------------------------------------------------------
# Mock models
# ---------------------------------------------------------------------------


class DeterministicModel(GenerativeModel):
    """Always predicts token 0 with probability 1 at every position."""

    def __init__(self):
        tok = MockTokenizer()
        super().__init__(
            model=nn.Linear(1, 1),
            tokenizer=tok,
            logit_formatter=PassThroughLogitFormatter(),
        )
        self._vocab_size = tok.VOCAB_SIZE

    def forward(self, seq_SP, **kwargs):
        S, P = seq_SP.shape
        logits = torch.full((S, P, self._vocab_size), -1e9)
        logits[:, :, 0] = 0.0
        return logits

    def format_raw_to_logits(self, raw, seq_SP, **kwargs):
        return raw.float()


class UniformModel(GenerativeModel):
    """Returns uniform probabilities over tokens 0-4 (5 'real' tokens).

    log p(any real token) = log(1/5) ≈ -1.609 at every position.
    """

    def __init__(self):
        tok = MockTokenizer()
        super().__init__(
            model=nn.Linear(1, 1),
            tokenizer=tok,
            logit_formatter=PassThroughLogitFormatter(),
        )
        self._vocab_size = tok.VOCAB_SIZE

    def forward(self, seq_SP, **kwargs):
        S, P = seq_SP.shape
        logits = torch.full((S, P, self._vocab_size), -1e9)
        logits[:, :, :5] = 0.0
        return logits

    def format_raw_to_logits(self, raw, seq_SP, **kwargs):
        return raw.float()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def det_model():
    return DeterministicModel()


@pytest.fixture
def uniform_model():
    return UniformModel()


@pytest.fixture
def tmp_plot(tmp_path):
    return tmp_path / "test_trajectory.png"


# ---------------------------------------------------------------------------
# compute_log_prob_trajectory — return structure
# ---------------------------------------------------------------------------


class TestReturnStructure:
    def test_returns_dict_with_expected_keys(self, det_model):
        result = compute_log_prob_trajectory(["01234"], det_model, n_time_points=3)
        assert "time_points" in result
        assert "avg_log_probs" in result

    def test_time_points_shape(self, det_model):
        n = 7
        result = compute_log_prob_trajectory(["01234"], det_model, n_time_points=n)
        assert result["time_points"].shape == (n,)

    def test_avg_log_probs_shape(self, det_model):
        seqs = ["01234", "01234", "01234"]
        n = 5
        result = compute_log_prob_trajectory(seqs, det_model, n_time_points=n)
        assert result["avg_log_probs"].shape == (3, n)

    def test_time_points_are_ascending(self, det_model):
        result = compute_log_prob_trajectory(["01234"], det_model, n_time_points=10)
        t = result["time_points"]
        assert torch.all(t[1:] > t[:-1])

    def test_time_points_start_at_zero(self, det_model):
        result = compute_log_prob_trajectory(["01234"], det_model, n_time_points=5)
        assert result["time_points"][0].item() == 0.0

    def test_time_points_below_one(self, det_model):
        result = compute_log_prob_trajectory(["01234"], det_model, n_time_points=5)
        assert result["time_points"][-1].item() < 1.0


# ---------------------------------------------------------------------------
# compute_log_prob_trajectory — log probability values
# ---------------------------------------------------------------------------


class TestLogProbValues:
    def test_all_log_probs_are_nonpositive(self, uniform_model):
        result = compute_log_prob_trajectory(
            ["01234", "43210"], uniform_model, n_time_points=5
        )
        lp = result["avg_log_probs"]
        assert torch.all(lp[~torch.isnan(lp)] <= 0.0 + 1e-6)

    def test_deterministic_model_log_probs_near_zero(self, det_model):
        """Sequences of all 0s: model predicts token 0 with p=1, so log p ≈ 0."""
        result = compute_log_prob_trajectory(["00000"], det_model, n_time_points=5)
        lp = result["avg_log_probs"]
        non_nan = lp[~torch.isnan(lp)]
        assert torch.allclose(non_nan, torch.zeros_like(non_nan), atol=1e-4)

    def test_uniform_model_log_probs_consistent(self, uniform_model):
        """UniformModel gives log(1/5) at every masked position."""
        expected = torch.tensor(1 / 5.0).log()
        result = compute_log_prob_trajectory(["01234"], uniform_model, n_time_points=5)
        lp = result["avg_log_probs"]
        non_nan = lp[~torch.isnan(lp)]
        assert torch.allclose(non_nan, expected.expand_as(non_nan), atol=1e-4)

    def test_at_t_zero_all_maskable_positions_masked(self, uniform_model):
        """At t=0, all maskable positions are masked → identical avg log prob."""
        seqs = ["01234", "43210", "22222"]
        result = compute_log_prob_trajectory(seqs, uniform_model, n_time_points=5)
        lp_at_t0 = result["avg_log_probs"][:, 0]
        assert torch.allclose(lp_at_t0, lp_at_t0[0].expand_as(lp_at_t0), atol=1e-4)


# ---------------------------------------------------------------------------
# compute_log_prob_trajectory — trajectory shape
# ---------------------------------------------------------------------------


class TestTrajectoryShape:
    def test_deterministic_model_flat_trajectory(self, det_model):
        """Token 0 always predicted with p=1 → flat at 0."""
        result = compute_log_prob_trajectory(["00000"], det_model, n_time_points=10)
        lp = result["avg_log_probs"]
        non_nan = lp[~torch.isnan(lp)]
        assert torch.allclose(non_nan, torch.zeros_like(non_nan), atol=1e-4)

    def test_uniform_model_flat_trajectory(self, uniform_model):
        """Context-independent model → flat at log(1/5)."""
        expected = torch.tensor(1 / 5.0).log()
        result = compute_log_prob_trajectory(["01234"], uniform_model, n_time_points=10)
        lp = result["avg_log_probs"]
        non_nan = lp[~torch.isnan(lp)]
        assert torch.allclose(non_nan, expected.expand_as(non_nan), atol=1e-4)


# ---------------------------------------------------------------------------
# compute_log_prob_trajectory — batching
# ---------------------------------------------------------------------------


class TestBatching:
    def test_batch_size_does_not_affect_results(self, uniform_model):
        """UniformModel is context-independent, so batch_size shouldn't matter.

        We fix the random seed so both calls get the same masks, making the
        outputs directly comparable.
        """
        seqs = ["01234", "43210", "11111"]
        torch.manual_seed(42)
        result_bs1 = compute_log_prob_trajectory(
            seqs, uniform_model, n_time_points=3, batch_size=1
        )
        torch.manual_seed(42)
        result_bs10 = compute_log_prob_trajectory(
            seqs, uniform_model, n_time_points=3, batch_size=10
        )
        lp1 = torch.nan_to_num(result_bs1["avg_log_probs"], nan=-999.0)
        lp10 = torch.nan_to_num(result_bs10["avg_log_probs"], nan=-999.0)
        assert torch.allclose(lp1, lp10, atol=1e-4)


# ---------------------------------------------------------------------------
# compute_log_prob_trajectory — edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_sequence(self, det_model):
        result = compute_log_prob_trajectory(["00"], det_model, n_time_points=3)
        assert result["avg_log_probs"].shape[0] == 1

    def test_single_time_point(self, det_model):
        result = compute_log_prob_trajectory(["01234"], det_model, n_time_points=1)
        assert result["time_points"].shape == (1,)
        assert result["avg_log_probs"].shape == (1, 1)

    def test_many_time_points(self, uniform_model):
        result = compute_log_prob_trajectory(["01234"], uniform_model, n_time_points=50)
        assert result["time_points"].shape == (50,)

    def test_variable_length_sequences(self, uniform_model):
        """Different-length sequences should work (tokenizer pads)."""
        seqs = ["012", "01234", "0"]
        result = compute_log_prob_trajectory(seqs, uniform_model, n_time_points=5)
        assert result["avg_log_probs"].shape[0] == 3
        lp = result["avg_log_probs"]
        non_nan = lp[~torch.isnan(lp)]
        expected = torch.tensor(1 / 5.0).log()
        assert torch.allclose(non_nan, expected.expand_as(non_nan), atol=1e-4)


# ---------------------------------------------------------------------------
# compute_decoding_log_prob_trajectory
# ---------------------------------------------------------------------------


class TestDecodingTrajectory:
    def test_returns_one_curve_per_sequence(self, uniform_model):
        seqs = ["01234", "43210"]
        orders = [torch.tensor([1, 2, 3, 4, 5]), torch.tensor([1, 2, 3, 4, 5])]
        result = compute_decoding_log_prob_trajectory(seqs, uniform_model, orders)

        assert len(result["percent_unmasked"]) == 2
        assert len(result["decoded_position_log_probs"]) == 2
        assert result["percent_unmasked"][0].shape == (5,)
        assert result["decoded_position_log_probs"][0].shape == (5,)

    def test_uniform_model_values_are_constant(self, uniform_model):
        seqs = ["01234"]
        orders = [torch.tensor([1, 2, 3, 4, 5])]
        result = compute_decoding_log_prob_trajectory(seqs, uniform_model, orders)
        expected = torch.tensor(1 / 5.0).log()
        lp = result["decoded_position_log_probs"][0]
        assert torch.allclose(lp, expected.expand_as(lp), atol=1e-4)

    def test_invalid_order_raises(self, uniform_model):
        seqs = ["01234"]
        orders = [torch.tensor([0, 1, 2, 3, 4])]  # includes BOS (non-maskable)
        with pytest.raises(ValueError):
            compute_decoding_log_prob_trajectory(seqs, uniform_model, orders)


# ---------------------------------------------------------------------------
# plot_log_prob_trajectories — file output
# ---------------------------------------------------------------------------


class TestPlotOutput:
    def test_plot_file_created(self, det_model, tmp_plot):
        traj = compute_log_prob_trajectory(["01234"], det_model, n_time_points=3)
        plot_log_prob_trajectories([traj], ["test"], tmp_plot)
        assert tmp_plot.exists()

    def test_plot_file_is_nonempty(self, det_model, tmp_plot):
        traj = compute_log_prob_trajectory(["01234"], det_model, n_time_points=3)
        plot_log_prob_trajectories([traj], ["test"], tmp_plot)
        assert tmp_plot.stat().st_size > 0

    def test_plot_creates_parent_directories(self, det_model, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "plot.png"
        traj = compute_log_prob_trajectory(["01234"], det_model, n_time_points=3)
        plot_log_prob_trajectories([traj], ["test"], nested)
        assert nested.exists()

    def test_plot_file_is_valid_png(self, det_model, tmp_plot):
        traj = compute_log_prob_trajectory(["01234"], det_model, n_time_points=3)
        plot_log_prob_trajectories([traj], ["test"], tmp_plot)
        with open(tmp_plot, "rb") as f:
            header = f.read(8)
        assert header[:4] == b"\x89PNG"


# ---------------------------------------------------------------------------
# plot_log_prob_trajectories — multi-condition
# ---------------------------------------------------------------------------


class TestMultiConditionPlot:
    def test_two_conditions(self, det_model, uniform_model, tmp_plot):
        traj1 = compute_log_prob_trajectory(["00000"], det_model, n_time_points=5)
        traj2 = compute_log_prob_trajectory(["01234"], uniform_model, n_time_points=5)
        plot_log_prob_trajectories([traj1, traj2], ["det", "uniform"], tmp_plot)
        assert tmp_plot.exists()
        assert tmp_plot.stat().st_size > 0

    def test_many_conditions(self, uniform_model, tmp_plot):
        trajs = []
        labels = []
        for i in range(5):
            traj = compute_log_prob_trajectory(
                ["01234"], uniform_model, n_time_points=3
            )
            trajs.append(traj)
            labels.append(f"condition_{i}")
        plot_log_prob_trajectories(trajs, labels, tmp_plot)
        assert tmp_plot.exists()

    def test_mismatched_labels_raises(self, det_model, tmp_plot):
        traj = compute_log_prob_trajectory(["01234"], det_model, n_time_points=3)
        with pytest.raises(AssertionError):
            plot_log_prob_trajectories([traj], ["a", "b"], tmp_plot)

    def test_no_individual_lines(self, det_model, tmp_plot):
        """show_individual=False should still produce a valid plot."""
        traj = compute_log_prob_trajectory(["01234"], det_model, n_time_points=3)
        plot_log_prob_trajectories([traj], ["test"], tmp_plot, show_individual=False)
        assert tmp_plot.exists()
        assert tmp_plot.stat().st_size > 0


class TestDecodingPlot:
    def test_decoding_plot_file_created(self, uniform_model, tmp_plot):
        seqs = ["01234", "43210"]
        orders = [torch.tensor([1, 2, 3, 4, 5]), torch.tensor([1, 2, 3, 4, 5])]
        traj = compute_decoding_log_prob_trajectory(seqs, uniform_model, orders)
        plot_decoding_log_prob_trajectories([traj], ["uniform"], tmp_plot)
        assert tmp_plot.exists()
        assert tmp_plot.stat().st_size > 0
