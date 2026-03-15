"""Integration tests for LoRA on ESMC model.

These tests load the real ESMC-300m model, so they are slower than the
unit tests in test_lora.py. They verify that LoRA works correctly through
the full ESMC pipeline: forward, get_log_probs, differentiable_embedding,
and checkpointing.
"""

import json
import torch
import pytest
from peft import PeftModel

from dfm.models.esm import ESMC
from dfm.predictive_modeling import LinearProbe, point_estimate_binary_logits


@pytest.fixture(scope="module")
def esmc():
    return ESMC("esmc_300m")


@pytest.fixture
def seq():
    """CLS + ACDE + EOS"""
    tok = ESMC("esmc_300m").tokenizer
    return tok(["ACDE"], return_tensors="pt")["input_ids"]


# ── Discovery ────────────────────────────────────────────────────────────────


class TestESMCLoraDiscovery:
    def test_lora_target_modules(self, esmc):
        targets = esmc.lora_target_modules()
        assert "transformer.blocks.*.attn.layernorm_qkv.1" in targets
        assert "transformer.blocks.*.attn.out_proj" in targets
        assert "transformer.blocks.*.ffn.1" in targets
        assert "transformer.blocks.*.ffn.3" in targets
        # QKV projection: 960 → 2880, 30 blocks
        info = targets["transformer.blocks.*.attn.layernorm_qkv.1"]
        assert info == (960, 2880, 30)


# ── Apply + Forward ─────────────────────────────────────────────────────────


class TestESMCLoraForward:
    def test_apply_lora_and_forward(self, seq):
        model = ESMC("esmc_300m")
        model.apply_lora(target_modules=["layernorm_qkv.1", "out_proj"], r=4)
        out = model(seq)
        # ESMC forward returns ESMCOutput dataclass, not raw tensor
        assert out.sequence_logits.shape == (1, 6, 64)

    def test_get_log_probs_with_lora(self, seq):
        model = ESMC("esmc_300m")
        model.apply_lora(target_modules=["layernorm_qkv.1", "out_proj"], r=4)
        log_probs = model.get_log_probs(seq)
        probs = log_probs.exp()
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_differentiable_embedding_with_lora(self, seq):
        model = ESMC("esmc_300m")
        model.apply_lora(target_modules=["layernorm_qkv.1", "out_proj"], r=4)
        emb = model.embed(seq)
        assert emb.shape == (1, 6, model.EMB_DIM)

    def test_lora_gradients_flow(self, seq):
        model = ESMC("esmc_300m")
        model.apply_lora(target_modules=["layernorm_qkv.1", "out_proj"], r=4)
        out = model(seq)
        loss = out.sequence_logits.sum()
        loss.backward()
        lora_grads = {
            n: p.grad for n, p in model.model.named_parameters()
            if "lora_" in n and p.grad is not None
        }
        assert len(lora_grads) > 0


# ── Checkpointing ───────────────────────────────────────────────────────────


class TestESMCCheckpointing:
    def test_save_args(self, esmc):
        args = esmc._save_args()
        assert args == {"esmc_checkpoint": "esmc_300m"}

    def test_round_trip_no_lora(self, tmp_path):
        model = ESMC("esmc_300m")
        model.save(tmp_path / "esmc")
        config = json.loads((tmp_path / "esmc" / "config.json").read_text())
        assert config == {"esmc_checkpoint": "esmc_300m"}
        assert not (tmp_path / "esmc" / "lora_adapter").exists()

        loaded = ESMC.from_checkpoint(tmp_path / "esmc")
        assert not loaded.has_lora
        assert loaded._esmc_checkpoint == "esmc_300m"

    def test_round_trip_with_lora(self, tmp_path, seq):
        model = ESMC("esmc_300m")
        model.apply_lora(target_modules=["layernorm_qkv.1", "out_proj"], r=4)

        # Train a step
        opt = torch.optim.SGD(
            [p for p in model.model.parameters() if p.requires_grad], lr=0.1
        )
        out = model(seq)
        out.sequence_logits.sum().backward()
        opt.step()
        out_trained = model(seq).sequence_logits.detach()

        model.save(tmp_path / "esmc")
        assert (tmp_path / "esmc" / "lora_adapter").exists()

        loaded = ESMC.from_checkpoint(tmp_path / "esmc")
        assert loaded.has_lora
        out_loaded = loaded(seq).sequence_logits.detach()
        assert torch.allclose(out_trained, out_loaded, atol=1e-5)


# ── LinearProbe + LoRA ───────────────────────────────────────────────────────


class TestLinearProbeWithLora:
    def test_freeze_embed_model_false_preserves_lora(self, seq):
        """With freeze_embed_model=False, LoRA params remain trainable."""
        esmc = ESMC("esmc_300m")
        esmc.apply_lora(target_modules=["layernorm_qkv.1", "out_proj"], r=4)

        class FitnessProbe(LinearProbe):
            def format_raw_to_logits(self, raw_output, ohe, **kwargs):
                return point_estimate_binary_logits(raw_output, self.target, k=10.0)

        probe = FitnessProbe(embed_model=esmc, output_dim=1, freeze_embed_model=False)
        probe.set_target_(0.5)

        # LoRA params should be trainable
        lora_params = [
            (n, p) for n, p in probe.embed_model.model.named_parameters()
            if "lora_" in n
        ]
        assert len(lora_params) > 0
        for name, p in lora_params:
            assert p.requires_grad, f"LoRA param {name} should be trainable"

        # Base params should be frozen
        base_params = [
            (n, p) for n, p in probe.embed_model.model.named_parameters()
            if "lora_" not in n
        ]
        for name, p in base_params:
            assert not p.requires_grad, f"Base param {name} should be frozen"

        # Head params should be trainable
        for name, p in probe.w.named_parameters():
            assert p.requires_grad, f"Head param {name} should be trainable"

    def test_freeze_embed_model_true_freezes_lora(self, seq):
        """With freeze_embed_model=True (default), ALL embed_model params are frozen."""
        esmc = ESMC("esmc_300m")
        esmc.apply_lora(target_modules=["layernorm_qkv.1", "out_proj"], r=4)

        class FitnessProbe(LinearProbe):
            def format_raw_to_logits(self, raw_output, ohe, **kwargs):
                return point_estimate_binary_logits(raw_output, self.target, k=10.0)

        probe = FitnessProbe(embed_model=esmc, output_dim=1, freeze_embed_model=True)

        for name, p in probe.embed_model.named_parameters():
            assert not p.requires_grad, f"Param {name} should be frozen"

    def test_end_to_end_forward(self, seq):
        """Full forward through LinearProbe with LoRA-adapted ESMC."""
        esmc = ESMC("esmc_300m")
        esmc.apply_lora(target_modules=["layernorm_qkv.1", "out_proj"], r=4)

        class FitnessProbe(LinearProbe):
            def format_raw_to_logits(self, raw_output, ohe, **kwargs):
                return point_estimate_binary_logits(raw_output, self.target, k=10.0)

        probe = FitnessProbe(embed_model=esmc, output_dim=1, freeze_embed_model=False)
        pred = probe.predict(seq)
        assert pred.shape == (1, 1)
