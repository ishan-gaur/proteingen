import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from ...predictive_modeling import PredictiveModel, binary_logits
from typing import TypedDict


# Helper functions


def gather_edges(edges, neighbor_idx):
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def gather_nodes_t(nodes, neighbor_idx):
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


# Model components


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h


class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(
            offset + self.max_relative_feature, 0, 2 * self.max_relative_feature
        ) * mask + (1 - mask) * (2 * self.max_relative_feature + 1)
        d_onehot = torch.nn.functional.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        E = self.linear(d_onehot.float())
        return E


class ProteinFeatures(nn.Module):
    def __init__(
        self,
        edge_features,
        node_features,
        num_positional_embeddings=16,
        num_rbf=16,
        top_k=30,
        augment_eps=0.0,
        num_chain_embeddings=16,
    ):
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        node_in, edge_in = 6, num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1e-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(
            D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False
        )
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, X, mask, residue_idx, chain_labels):
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]
        D_neighbors, E_idx = self._dist(Ca, mask)
        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))
        RBF_all.append(self._get_rbf(N, N, E_idx))
        RBF_all.append(self._get_rbf(C, C, E_idx))
        RBF_all.append(self._get_rbf(O, O, E_idx))
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx))
        RBF_all.append(self._get_rbf(Ca, N, E_idx))
        RBF_all.append(self._get_rbf(Ca, C, E_idx))
        RBF_all.append(self._get_rbf(Ca, O, E_idx))
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx))
        RBF_all.append(self._get_rbf(N, C, E_idx))
        RBF_all.append(self._get_rbf(N, O, E_idx))
        RBF_all.append(self._get_rbf(N, Cb, E_idx))
        RBF_all.append(self._get_rbf(Cb, C, E_idx))
        RBF_all.append(self._get_rbf(Cb, O, E_idx))
        RBF_all.append(self._get_rbf(O, C, E_idx))
        RBF_all.append(self._get_rbf(N, Ca, E_idx))
        RBF_all.append(self._get_rbf(C, Ca, E_idx))
        RBF_all.append(self._get_rbf(O, Ca, E_idx))
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx))
        RBF_all.append(self._get_rbf(C, N, E_idx))
        RBF_all.append(self._get_rbf(O, N, E_idx))
        RBF_all.append(self._get_rbf(Cb, N, E_idx))
        RBF_all.append(self._get_rbf(C, Cb, E_idx))
        RBF_all.append(self._get_rbf(O, Cb, E_idx))
        RBF_all.append(self._get_rbf(C, O, E_idx))
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)
        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]
        d_chains = ((chain_labels[:, :, None] - chain_labels[:, None, :]) == 0).long()
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx


class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)
        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class FlowMatchPMPNN(nn.Module):
    def __init__(
        self,
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=4,
        num_decoder_layers=4,
        vocab=22,
        k_neighbors=32,
        augment_eps=0.1,
        dropout=0.1,
    ):
        super(FlowMatchPMPNN, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab
        self.features = ProteinFeatures(
            node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps
        )
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)
        self.encoder_layers = nn.ModuleList(
            [
                EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout)
                for _ in range(num_decoder_layers)
            ]
        )
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        h_V, h_E, E_idx, mask_attend = self.encode_structure(
            X, mask, chain_M, residue_idx, chain_encoding_all
        )
        log_probs, logits = self.decode(h_V, h_E, E_idx, S, mask_attend, mask, chain_M)
        return log_probs, logits

    def encode_structure(self, X, mask, chain_M, residue_idx, chain_encoding_all):
        device = X.device
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = torch.utils.checkpoint.checkpoint(
                layer, h_V, h_E, E_idx, mask, mask_attend, use_reentrant=False
            )
        return h_V, h_E, E_idx, mask_attend

    def decode(self, h_V, h_E, E_idx, S, mask_attend, mask, chain_M):
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        chain_M = chain_M * mask
        mask_size = E_idx.shape[1]
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D
        mask_fw = mask_1D
        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_V = torch.utils.checkpoint.checkpoint(
                layer, h_V, h_ESV, mask, use_reentrant=False
            )
        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, logits


class StabilityPMPNN(nn.Module):
    @classmethod
    def init(
        cls,
        num_letters=21,
        vocab=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=4,
        num_decoder_layers=4,
        k_neighbors=30,
        dropout=0.0,
        augment_eps=0.0,
    ):
        fm_mpnn = FlowMatchPMPNN(
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            k_neighbors=k_neighbors,
            dropout=dropout,
            augment_eps=augment_eps,
            vocab=vocab,
            num_letters=num_letters,
        )
        return cls(fm_mpnn)

    def __init__(self, fm_mpnn):
        super(StabilityPMPNN, self).__init__()
        self.fm_mpnn = fm_mpnn
        self.lin1 = nn.Linear(128, 128)
        self.act = nn.ReLU()
        self.lin2 = nn.Linear(128, 1)

    def encode_structure(self, X, mask, chain_M, residue_idx, chain_encoding_all):
        h_V, h_E, E_idx, mask_attend = self.fm_mpnn.encode_structure(
            X, mask, chain_M, residue_idx, chain_encoding_all
        )
        return h_V, h_E, E_idx, mask_attend

    def decode_to_hidden(self, h_V, h_E, E_idx, S, mask_attend, mask, chain_M):
        h_S = self.fm_mpnn.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        chain_M = chain_M * mask
        mask_size = E_idx.shape[1]
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D
        mask_fw = mask_1D
        for layer in self.fm_mpnn.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_V = torch.utils.checkpoint.checkpoint(
                layer, h_V, h_ESV, mask, use_reentrant=False
            )
        return h_V * mask.unsqueeze(-1)

    def decode(self, h_V, h_E, E_idx, S, mask_attend, mask, chain_M):
        h_V = self.decode_to_hidden(h_V, h_E, E_idx, S, mask_attend, mask, chain_M)
        h_G = h_V.mean(dim=1)
        h_G = self.act(self.lin1(h_G))
        h_G = self.lin2(h_G)
        return h_G

    def get_hidden(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        h_V, h_E, E_idx, mask_attend = self.encode_structure(
            X, mask, chain_M, residue_idx, chain_encoding_all
        )
        h_V = self.decode_to_hidden(h_V, h_E, E_idx, S, mask_attend, mask, chain_M)
        return h_V

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        h_V = self.get_hidden(X, S, mask, chain_M, residue_idx, chain_encoding_all)
        h_G = h_V.mean(dim=1)
        h_G = self.act(self.lin1(h_G))
        h_G = self.lin2(h_G)
        return h_G


class StabilityPredictorConditioning(TypedDict):
    X: torch.Tensor
    mask: torch.Tensor
    chain_M: torch.Tensor
    residue_idx: torch.Tensor
    chain_encoding_all: torch.Tensor


class PMPNNStructureEncoding(TypedDict):
    h_V: torch.Tensor
    h_E: torch.Tensor
    E_idx: torch.Tensor
    mask_attend: torch.Tensor
    mask: torch.Tensor
    chain_M: torch.Tensor


class PreTrainedStabilityPredictor(PredictiveModel):
    """Binary stability predictor wrapping a StabilityPMPNN.

    The model outputs a single logit per sequence where sigmoid(logit) = P(stable).
    Uses ``binary_logits`` for the binary logit conversion via
    ``sigmoid(x) = softmax([0, x])[1]``.

    Target defaults to ``True`` (P(stable)). Set to ``False`` for P(not stable).
    """

    def __init__(
        self, ckpt_path, vocab_size=21, device="cuda", one_hot_encode_input=False
    ):
        from ...generative_modeling import MPNNTokenizer

        tokenizer = MPNNTokenizer(include_mask_token=True)
        super().__init__(tokenizer=tokenizer)
        self.target = True
        self.stability_model = StabilityPMPNN.init(num_letters=vocab_size, vocab=vocab_size)
        self.stability_model.load_state_dict(torch.load(ckpt_path, weights_only=False))
        if one_hot_encode_input:
            layer = nn.Linear(vocab_size, 128, bias=False)
            layer.weight.data = self.stability_model.fm_mpnn.W_s.weight.data.T.clone()
            self.stability_model.fm_mpnn.W_s = layer
        self.input_dim = vocab_size

        if self.tokenizer.vocab_size != self.input_dim + 1:
            raise ValueError(
                "Expected stability tokenizer vocab_size to be input_dim + 1 (extra <mask> token)"
            )
        token_ohe_basis = torch.zeros(self.tokenizer.vocab_size, self.input_dim)
        token_ohe_basis[: self.input_dim, : self.input_dim] = torch.eye(self.input_dim)
        self.register_buffer("_token_ohe_basis_TK", token_ohe_basis)

    def token_ohe_basis(self) -> torch.FloatTensor:
        """Map PMPNN token IDs to predictor OHE features.

        The extra tokenizer-level <mask> token maps to the all-zero vector, so
        TAG can pass explicit masks while preserving the legacy stability model's
        masking semantics.
        """
        return self._token_ohe_basis_TK

    @staticmethod
    def prepare_conditioning(pdb_path, device="cpu") -> StabilityPredictorConditioning:
        """Load PDB and build conditioning tensors (batch_size=1)."""
        from .data_utils import load_pdb_to_graph_dict, featurize

        graph = load_pdb_to_graph_dict(pdb_path)
        X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = (
            featurize([graph], device)
        )
        return StabilityPredictorConditioning(
            X=X,
            mask=mask,
            chain_M=chain_M,
            residue_idx=residue_idx,
            chain_encoding_all=chain_encoding_all,
        )

    def preprocess_observations(
        self, observations: StabilityPredictorConditioning
    ) -> PMPNNStructureEncoding:
        device = self.device
        obs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in observations.items()
        }
        with torch.no_grad():
            h_V, h_E, E_idx, mask_attend = self.stability_model.encode_structure(
                obs["X"],
                obs["mask"],
                obs["chain_M"],
                obs["residue_idx"],
                obs["chain_encoding_all"],
            )
        return PMPNNStructureEncoding(
            h_V=h_V,
            h_E=h_E,
            E_idx=E_idx,
            mask_attend=mask_attend,
            mask=obs["mask"],
            chain_M=obs["chain_M"],
        )

    def collate_observations(
        self, x_B: torch.Tensor, observations: PMPNNStructureEncoding
    ) -> dict:
        B = x_B.shape[0]
        return dict(
            h_V=observations["h_V"][0:1].expand(B, -1, -1),
            h_E=observations["h_E"][0:1].expand(B, -1, -1, -1),
            E_idx=observations["E_idx"][0:1].expand(B, -1, -1),
            mask_attend=observations["mask_attend"][0:1].expand(B, -1, -1),
            mask=observations["mask"][0:1].expand(B, -1),
            chain_M=observations["chain_M"][0:1].expand(B, -1),
        )

    def forward(self, x_B, *, h_V, h_E, E_idx, mask_attend, mask, chain_M):
        logit = self.stability_model.decode(h_V, h_E, E_idx, x_B, mask_attend, mask, chain_M)
        return logit

    def format_raw_to_logits(self, raw_output, x_B, **kwargs):
        return binary_logits(raw_output, self.target)
