"""XGBoost-based predictive model for protein fitness prediction.

Wraps XGBoost regressors/classifiers into the PredictiveModel framework,
enabling their use with DEG guidance. XGBoost models are non-differentiable,
so only DEG (enumeration-based) guidance works — not TAG (gradient-based).

The model operates on one-hot encoded sequences, flattened across positions,
matching the input format of OneHotMLP.
"""

from __future__ import annotations

from abc import ABC
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerBase

from proteingen.predictive_modeling import PredictiveModel


class XGBoostPredictor(PredictiveModel, ABC):
    """XGBoost model operating on one-hot encoded sequences.

    Receives OHE input from PredictiveModel.get_log_probs, flattens across
    all positions, and runs XGBoost prediction. Subclasses implement
    format_raw_to_logits to convert the raw output to binary logits.

    Since XGBoost is non-differentiable, grad_log_prob will raise an error.
    Use DEG guidance instead of TAG.

    Tensor Dimension Labels:
        S: batch (sample) index
        P: position in sequence
        T: token dimension (one-hot / vocab size)
        O: output dimension
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        output_dim: int = 1,
        **xgb_kwargs,
    ):
        super().__init__(tokenizer)
        self.output_dim = output_dim
        self.xgb_kwargs = xgb_kwargs
        self._model = None  # set after fit()

    @property
    def model(self):
        assert self._model is not None, (
            "XGBoost model not trained yet. Call fit() first."
        )
        return self._model

    def fit(
        self,
        ohe_sequences_SNF: torch.FloatTensor,
        labels_SO: torch.FloatTensor,
        eval_ohe: torch.FloatTensor | None = None,
        eval_labels: torch.FloatTensor | None = None,
    ) -> dict:
        """Train the XGBoost model on flattened OHE features.

        Args:
            ohe_sequences_SNF: One-hot encoded sequences, shape (N, P*T) — already flattened.
            labels_SO: Target labels, shape (N, O).
            eval_ohe: Optional validation OHE features for early stopping.
            eval_labels: Optional validation labels.

        Returns:
            Dict with training info (e.g. best iteration).
        """
        import xgboost as xgb

        X_train = ohe_sequences_SNF.numpy()
        y_train = labels_SO.numpy()

        if self.output_dim == 1:
            y_train = y_train.ravel()

        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_estimators": 500,
            "learning_rate": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            **self.xgb_kwargs,
        }

        if self.output_dim > 1:
            # Multi-output: train separate models
            self._model = []
            info = {}
            for o in range(self.output_dim):
                model_o = xgb.XGBRegressor(**params)
                eval_set = None
                if eval_ohe is not None and eval_labels is not None:
                    eval_set = [(eval_ohe.numpy(), eval_labels[:, o].numpy())]
                model_o.fit(
                    X_train,
                    y_train[:, o] if y_train.ndim > 1 else y_train,
                    eval_set=eval_set,
                    verbose=False,
                )
                self._model.append(model_o)
                info[f"output_{o}_n_estimators"] = params["n_estimators"]
            return info
        else:
            self._model = xgb.XGBRegressor(**params)
            eval_set = None
            if eval_ohe is not None and eval_labels is not None:
                eval_set = [(eval_ohe.numpy(), eval_labels.ravel().numpy())]
            self._model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
            return {"n_estimators": params["n_estimators"]}

    def forward(self, ohe_seq_SPT: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        """Forward pass: flatten OHE and run XGBoost prediction."""
        x_flat = ohe_seq_SPT.reshape(ohe_seq_SPT.size(0), -1)
        x_np = x_flat.detach().cpu().numpy()

        if isinstance(self._model, list):
            # Multi-output
            preds = []
            for model_o in self._model:
                preds.append(torch.tensor(model_o.predict(x_np), dtype=torch.float32))
            return torch.stack(preds, dim=-1).to(ohe_seq_SPT.device)
        else:
            pred = torch.tensor(self.model.predict(x_np), dtype=torch.float32)
            if pred.ndim == 1:
                pred = pred.unsqueeze(-1)
            return pred.to(ohe_seq_SPT.device)

    def predict(self, seq_SP: torch.LongTensor) -> torch.FloatTensor:
        """Get raw model predictions from token IDs."""
        ohe = self.tokens_to_ohe(seq_SP).float()
        return self.forward(ohe)

    def grad_log_prob(self, seq_SP: torch.LongTensor) -> torch.FloatTensor:
        raise NotImplementedError(
            "XGBoost is non-differentiable. Use DEG guidance instead of TAG."
        )

    def save_model(self, path: str | Path) -> None:
        """Save the trained XGBoost model(s) to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if isinstance(self._model, list):
            for i, m in enumerate(self._model):
                m.save_model(str(path / f"model_{i}.json"))
        else:
            self.model.save_model(str(path / "model.json"))

    def load_model(self, path: str | Path) -> None:
        """Load trained XGBoost model(s) from disk."""
        import xgboost as xgb

        path = Path(path)
        if (path / "model.json").exists():
            self._model = xgb.XGBRegressor()
            self._model.load_model(str(path / "model.json"))
        else:
            self._model = []
            i = 0
            while (path / f"model_{i}.json").exists():
                m = xgb.XGBRegressor()
                m.load_model(str(path / f"model_{i}.json"))
                self._model.append(m)
                i += 1
            assert len(self._model) > 0, f"No XGBoost model files found in {path}"
