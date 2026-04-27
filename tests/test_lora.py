"""Tests for LoRA adapter support on GenerativeModel."""

import json
import torch
import pytest
from torch import nn
from peft import PeftModel

from proteingen.modeling import (
    GenerativeModel,
    GenerativeModelWithEmbedding,
    PassThroughLogitFormatter,
)
from proteingen.modeling import ProbabilityModel
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer


# ── Fixtures ─────────────────────────────────────────────────────────────────

VOCAB_SIZE = 33
OUTPUT_DIM = 64


class TinyTransformer(nn.Module):
    """Minimal model with Linear layers to test LoRA injection."""

    def __init__(self, dim: int = 32, output_dim: int = OUTPUT_DIM):
        super().__init__()
        self.embed = nn.Embedding(OUTPUT_DIM, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim),
        )
        self.head = nn.Linear(dim, output_dim)

    def forward(self, seq_SP, **kwargs):
        x = self.embed(seq_SP)
        q = self.q_proj(x)
        v = self.v_proj(x)
        x = self.out_proj(q + v)
        x = self.ffn(x)
        return self.head(x)


@pytest.fixture
def tokenizer():
    return EsmSequenceTokenizer()


@pytest.fixture
def tiny_model(tokenizer):
    backbone = TinyTransformer()
    fmt = PassThroughLogitFormatter()
    return GenerativeModel(backbone, tokenizer, fmt)


@pytest.fixture
def seq():
    return torch.tensor([[0, 5, 6, 7, 2]])  # CLS, A, C, D, EOS


# ── lora_target_modules tests ────────────────────────────────────────────────


class TestLoraTargetModules:
    def test_returns_dict(self, tiny_model):
        targets = tiny_model.lora_target_modules()
        assert isinstance(targets, dict)

    def test_finds_all_linear_layers(self, tiny_model):
        targets = tiny_model.lora_target_modules()
        # TinyTransformer has: q_proj, v_proj, out_proj, ffn.0, ffn.2, head
        assert len(targets) == 6

    def test_shape_info_correct(self, tiny_model):
        targets = tiny_model.lora_target_modules()
        # q_proj is Linear(32, 32)
        assert targets["q_proj"] == (32, 32, 1)
        # ffn.0 is Linear(32, 128)
        assert targets["ffn.0"] == (32, 128, 1)

    def test_collapses_repeated_blocks(self, tokenizer):
        """Block indices like .0. .1. should be collapsed to .*."""

        class RepeatedBlocks(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([
                    nn.ModuleDict({"attn": nn.Linear(16, 16), "ffn": nn.Linear(16, 16)})
                    for _ in range(4)
                ])
                self._dummy = nn.Linear(1, 1)

            def forward(self, x, **kwargs):
                return x

        model = GenerativeModel(RepeatedBlocks(), tokenizer, PassThroughLogitFormatter())
        targets = model.lora_target_modules()
        # blocks.*.attn, blocks.*.ffn collapsed to 2 patterns + _dummy
        assert "blocks.*.attn" in targets
        assert "blocks.*.ffn" in targets
        assert targets["blocks.*.attn"] == (16, 16, 4)


# ── apply_lora tests ────────────────────────────────────────────────────────


class TestApplyLora:
    def test_applies_lora(self, tiny_model):
        assert not tiny_model.has_lora
        tiny_model.apply_lora(target_modules=["q_proj", "v_proj"], r=4)
        assert tiny_model.has_lora
        assert isinstance(tiny_model.model, PeftModel)

    def test_base_params_frozen(self, tiny_model):
        tiny_model.apply_lora(target_modules=["q_proj"], r=4)
        for name, p in tiny_model.model.named_parameters():
            if "lora_" not in name:
                assert not p.requires_grad, f"Base param {name} should be frozen"

    def test_lora_params_trainable(self, tiny_model):
        tiny_model.apply_lora(target_modules=["q_proj"], r=4)
        lora_params = [
            (n, p) for n, p in tiny_model.model.named_parameters()
            if "lora_" in n
        ]
        assert len(lora_params) > 0
        for name, p in lora_params:
            assert p.requires_grad, f"LoRA param {name} should be trainable"

    def test_forward_still_works(self, tiny_model, seq):
        tiny_model.apply_lora(target_modules=["q_proj", "v_proj"], r=4)
        out = tiny_model(seq)
        assert out.shape == (1, 5, OUTPUT_DIM)

    def test_get_log_probs_still_works(self, tiny_model, seq):
        tiny_model.apply_lora(target_modules=["q_proj"], r=4)
        log_probs = tiny_model.get_log_probs(seq)
        probs = log_probs.exp()
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_auto_target_all_linear(self, tiny_model):
        """When target_modules=None, targets all Linear layers."""
        tiny_model.apply_lora(r=4)
        assert tiny_model.has_lora
        # Check that LoRA params exist (targeting all linear layers)
        lora_params = [n for n, _ in tiny_model.model.named_parameters() if "lora_" in n]
        assert len(lora_params) > 0

    def test_custom_rank(self, tiny_model):
        tiny_model.apply_lora(target_modules=["q_proj"], r=2)
        for n, p in tiny_model.model.named_parameters():
            if "lora_A" in n:
                assert p.shape[0] == 2  # rank = 2

    def test_gradients_flow_through_lora(self, tiny_model, seq):
        tiny_model.apply_lora(target_modules=["q_proj", "v_proj"], r=4)
        out = tiny_model(seq)
        loss = out.sum()
        loss.backward()
        for n, p in tiny_model.model.named_parameters():
            if "lora_" in n and p.requires_grad:
                assert p.grad is not None, f"No gradient for {n}"


# ── has_lora tests ───────────────────────────────────────────────────────────


class TestHasLora:
    def test_false_initially(self, tiny_model):
        assert not tiny_model.has_lora

    def test_true_after_apply(self, tiny_model):
        tiny_model.apply_lora(target_modules=["q_proj"], r=4)
        assert tiny_model.has_lora


# ── save/load LoRA tests ────────────────────────────────────────────────────


class TestSaveLoadLora:
    def test_save_lora_creates_files(self, tiny_model, tmp_path):
        tiny_model.apply_lora(target_modules=["q_proj", "v_proj"], r=4)
        tiny_model.save_lora(tmp_path / "adapter")
        assert (tmp_path / "adapter" / "adapter_config.json").exists()

    def test_save_lora_requires_lora(self, tiny_model, tmp_path):
        with pytest.raises(AssertionError, match="No LoRA adapter"):
            tiny_model.save_lora(tmp_path / "adapter")

    def test_load_lora_restores_adapter(self, tokenizer, tmp_path, seq):
        # Use seeded base weights so both models match
        torch.manual_seed(42)
        backbone1 = TinyTransformer()
        m = GenerativeModel(backbone1, tokenizer, PassThroughLogitFormatter())

        m.apply_lora(target_modules=["q_proj", "v_proj"], r=4)
        out_before = m(seq).detach()

        opt = torch.optim.SGD(
            [p for p in m.model.parameters() if p.requires_grad], lr=1.0
        )
        loss = m(seq).sum()
        loss.backward()
        opt.step()
        out_after_train = m(seq).detach()
        assert not torch.allclose(out_before, out_after_train)

        m.save_lora(tmp_path / "adapter")

        # Load onto a fresh model with same base weights
        torch.manual_seed(42)
        backbone2 = TinyTransformer()
        fresh = GenerativeModel(backbone2, tokenizer, PassThroughLogitFormatter())
        fresh.load_lora(tmp_path / "adapter")
        out_loaded = fresh(seq).detach()
        assert torch.allclose(out_after_train, out_loaded, atol=1e-5)

    def test_load_lora_rejects_double_apply(self, tiny_model, tmp_path):
        tiny_model.apply_lora(target_modules=["q_proj"], r=4)
        tiny_model.save_lora(tmp_path / "adapter")
        with pytest.raises(AssertionError, match="already has a LoRA"):
            tiny_model.load_lora(tmp_path / "adapter")


# ── Checkpointing tests (ProbabilityModel / GenerativeModel) ────────────────


class TestCheckpointing:
    """Test the save/from_checkpoint protocol."""

    def test_probability_model_save_args_not_implemented(self):
        """Base ProbabilityModel._save_args raises."""

        class Bare(ProbabilityModel):
            def __init__(self):
                super().__init__()
                self._dummy = nn.Linear(1, 1)

            def forward(self, x, **kwargs):
                return x

            def format_raw_to_logits(self, raw, x, **kwargs):
                return raw

        m = Bare()
        with pytest.raises(NotImplementedError):
            m._save_args()

    def test_generative_model_save_creates_config(self, tmp_path, tokenizer):
        """A GenerativeModel subclass with _save_args can save."""

        class MyModel(GenerativeModel):
            def __init__(self, dim: int = 32):
                self._dim = dim
                backbone = TinyTransformer(dim=dim)
                super().__init__(backbone, EsmSequenceTokenizer(), PassThroughLogitFormatter())

            def _save_args(self):
                return {"dim": self._dim}

        m = MyModel(dim=16)
        m.save(tmp_path / "model")
        assert (tmp_path / "model" / "config.json").exists()
        config = json.loads((tmp_path / "model" / "config.json").read_text())
        assert config == {"dim": 16}

    def test_generative_model_from_checkpoint_round_trip(self, tmp_path):
        class MyModel(GenerativeModel):
            def __init__(self, dim: int = 32):
                self._dim = dim
                backbone = TinyTransformer(dim=dim)
                super().__init__(backbone, EsmSequenceTokenizer(), PassThroughLogitFormatter())

            def _save_args(self):
                return {"dim": self._dim}

        m = MyModel(dim=16)
        m.save(tmp_path / "model")
        loaded = MyModel.from_checkpoint(tmp_path / "model")
        assert loaded._dim == 16

    def test_generative_model_save_with_lora(self, tmp_path, tokenizer):
        class MyModel(GenerativeModel):
            def __init__(self, dim: int = 32):
                self._dim = dim
                backbone = TinyTransformer(dim=dim)
                super().__init__(backbone, EsmSequenceTokenizer(), PassThroughLogitFormatter())

            def _save_args(self):
                return {"dim": self._dim}

        m = MyModel(dim=32)
        m.apply_lora(target_modules=["q_proj"], r=4)
        m.save(tmp_path / "model")
        assert (tmp_path / "model" / "config.json").exists()
        assert (tmp_path / "model" / "lora_adapter").exists()

    def test_generative_model_from_checkpoint_with_lora(self, tmp_path, seq):
        class MyModel(GenerativeModel):
            def __init__(self, dim: int = 32, seed: int = 0):
                self._dim = dim
                self._seed = seed
                torch.manual_seed(seed)
                backbone = TinyTransformer(dim=dim)
                super().__init__(backbone, EsmSequenceTokenizer(), PassThroughLogitFormatter())

            def _save_args(self):
                return {"dim": self._dim, "seed": self._seed}

        m = MyModel(dim=32, seed=42)
        m.apply_lora(target_modules=["q_proj", "v_proj"], r=4)

        # Train a step
        opt = torch.optim.SGD(
            [p for p in m.model.parameters() if p.requires_grad], lr=1.0
        )
        loss = m(seq).sum()
        loss.backward()
        opt.step()
        out_trained = m(seq).detach()

        m.save(tmp_path / "model")

        loaded = MyModel.from_checkpoint(tmp_path / "model")
        assert loaded.has_lora
        out_loaded = loaded(seq).detach()
        assert torch.allclose(out_trained, out_loaded, atol=1e-5)
