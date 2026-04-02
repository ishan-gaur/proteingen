"""Tests for XGBoostPredictor."""

import pytest
import torch
from types import SimpleNamespace

from proteingen.models.xgboost_model import XGBoostPredictor
from proteingen.predictive_modeling import point_estimate_binary_logits


class DummyXGBoostPredictor(XGBoostPredictor):
    """Concrete subclass for testing."""

    def __init__(self, **kwargs):
        tokenizer = SimpleNamespace(
            vocab_size=21,
            pad_token_id=20,
            mask_token_id=None,
            unk_token_id=None,
            cls_token_id=None,
            eos_token_id=None,
            added_tokens_decoder={},
            vocab={chr(65 + i): i for i in range(20)} | {"<pad>": 20},
        )
        super().__init__(tokenizer=tokenizer, output_dim=1, **kwargs)

    def format_raw_to_logits(self, raw_output, ohe_seq_SPT, **kwargs):
        return point_estimate_binary_logits(raw_output.squeeze(-1), self.target)


class TestXGBoostPredictor:
    def test_construction(self):
        model = DummyXGBoostPredictor()
        assert model.output_dim == 1
        assert model._model is None

    def test_fit_and_predict(self):
        model = DummyXGBoostPredictor(n_estimators=10, max_depth=3)

        # Create dummy data: 50 sequences, length 5, vocab 21
        N, L, V = 50, 5, 21
        tokens = torch.randint(0, 20, (N, L))
        labels = torch.randn(N, 1)

        ohe = torch.nn.functional.one_hot(tokens, V).float().reshape(N, -1)
        model.fit(ohe, labels)

        # Predict
        ohe_3d = ohe.reshape(N, L, V)
        preds = model.forward(ohe_3d)
        assert preds.shape == (N, 1)

    def test_predict_from_tokens(self):
        model = DummyXGBoostPredictor(n_estimators=10, max_depth=3)

        N, L, V = 50, 5, 21
        tokens = torch.randint(0, 20, (N, L))
        labels = torch.randn(N, 1)

        ohe = torch.nn.functional.one_hot(tokens, V).float().reshape(N, -1)
        model.fit(ohe, labels)

        # predict() should work from token IDs
        preds = model.predict(tokens[:5])
        assert preds.shape == (5, 1)

    def test_multi_output(self):
        tokenizer = SimpleNamespace(
            vocab_size=21,
            pad_token_id=20,
            mask_token_id=None,
            unk_token_id=None,
            cls_token_id=None,
            eos_token_id=None,
            added_tokens_decoder={},
            vocab={chr(65 + i): i for i in range(20)} | {"<pad>": 20},
        )

        class MultiOutputXGB(XGBoostPredictor):
            def format_raw_to_logits(self, raw_output, ohe_seq_SPT, **kwargs):
                raise NotImplementedError

        model = MultiOutputXGB(
            tokenizer=tokenizer, output_dim=2, n_estimators=10, max_depth=3
        )

        N, L, V = 50, 5, 21
        tokens = torch.randint(0, 20, (N, L))
        labels = torch.randn(N, 2)

        ohe = torch.nn.functional.one_hot(tokens, V).float().reshape(N, -1)
        model.fit(ohe, labels)

        ohe_3d = ohe.reshape(N, L, V)
        preds = model.forward(ohe_3d)
        assert preds.shape == (N, 2)

    def test_grad_log_prob_raises(self):
        model = DummyXGBoostPredictor(n_estimators=10, max_depth=3)

        N, L, V = 50, 5, 21
        tokens = torch.randint(0, 20, (N, L))
        labels = torch.randn(N, 1)
        ohe = torch.nn.functional.one_hot(tokens, V).float().reshape(N, -1)
        model.fit(ohe, labels)

        with pytest.raises(NotImplementedError, match="non-differentiable"):
            model.grad_log_prob(tokens[:5])

    def test_get_log_probs_with_target(self):
        model = DummyXGBoostPredictor(n_estimators=10, max_depth=3)

        N, L, V = 50, 5, 21
        tokens = torch.randint(0, 20, (N, L))
        labels = torch.randn(N, 1)
        ohe = torch.nn.functional.one_hot(tokens, V).float().reshape(N, -1)
        model.fit(ohe, labels)

        model.set_target_(0.0)
        log_probs = model.get_log_probs(tokens[:5])
        assert log_probs.shape == (5,)
        assert (log_probs <= 0).all()  # log probs should be negative

    def test_save_and_load(self, tmp_path):
        model = DummyXGBoostPredictor(n_estimators=10, max_depth=3)

        N, L, V = 50, 5, 21
        tokens = torch.randint(0, 20, (N, L))
        labels = torch.randn(N, 1)
        ohe = torch.nn.functional.one_hot(tokens, V).float().reshape(N, -1)
        model.fit(ohe, labels)

        # Save
        model.save_model(tmp_path / "xgb")

        # Load into new model
        model2 = DummyXGBoostPredictor()
        model2.load_model(tmp_path / "xgb")

        # Predictions should match
        ohe_3d = ohe[:5].reshape(5, L, V)
        p1 = model.forward(ohe_3d)
        p2 = model2.forward(ohe_3d)
        torch.testing.assert_close(p1, p2)
