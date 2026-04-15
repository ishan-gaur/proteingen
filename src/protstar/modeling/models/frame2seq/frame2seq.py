from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import torch
from torch import nn
from torch.nn import functional as F

from frame2seq import __file__ as _frame2seq_init_file
from frame2seq.model.Frame2seq import frame2seq as _Frame2seqModel
from frame2seq.utils.featurize import make_s_init, make_z_init
from frame2seq.utils.pdb2input import get_inference_inputs
from frame2seq.utils.rigid_utils import Rigid

from ...generative_modeling import (
    GenerativeModelWithEmbedding,
    LogitFormatter,
)


class Frame2seqConditioning(TypedDict):
    pdb_path: str | Path
    chain_id: str


class _Frame2seqCachedConditioning(TypedDict):
    X_LA3: torch.FloatTensor
    seq_mask_L: torch.BoolTensor
    native_seq_L: torch.LongTensor


class Frame2seqTokenizer:
    """Tokenizer for Frame2seq's 21-token amino-acid vocabulary.

    Token Index Legend:
        AAs: 0-19 (A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y)
        X: 20 (unknown)
        <mask>: 21 (input-only mask token used by protstar)
    """

    _AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

    def __init__(self, include_mask_token: bool = True, mask_token: str = "<mask>"):
        self._vocab = {aa: i for i, aa in enumerate(self._AA_ORDER)}
        self._vocab["X"] = 20

        self.unk_token = "X"
        self.unk_token_id = self._vocab[self.unk_token]

        self.mask_token = None
        self.mask_token_id = None
        if include_mask_token:
            self.mask_token = mask_token
            self.mask_token_id = len(self._vocab)
            self._vocab[mask_token] = self.mask_token_id

        self._idx_to_token = {idx: token for token, idx in self._vocab.items()}

        self.cls_token_id = None
        self.eos_token_id = None
        self.pad_token_id = self.mask_token_id
        self.added_tokens_decoder = (
            {self.mask_token_id: self.mask_token}
            if self.mask_token_id is not None
            else {}
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    @property
    def all_special_ids(self) -> list[int]:
        if self.mask_token_id is None:
            return []
        return [self.mask_token_id]

    def encode(self, sequence: str) -> list[int]:
        return [self._vocab.get(aa, self.unk_token_id) for aa in sequence]

    def decode(
        self,
        token_ids: list[int] | torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        decoded_tokens = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in self.all_special_ids:
                continue
            decoded_tokens.append(self._idx_to_token.get(token_id, self.unk_token))
        return "".join(decoded_tokens)

    def __call__(
        self,
        sequences: str | list[str],
        padding: bool = False,
        return_tensors: str | None = None,
    ) -> dict[str, list[list[int]] | torch.LongTensor]:
        if isinstance(sequences, str):
            sequences = [sequences]

        encoded = [self.encode(seq) for seq in sequences]

        if padding:
            assert self.pad_token_id is not None
            max_len = max(len(ids) for ids in encoded)
            encoded = [
                ids + [self.pad_token_id] * (max_len - len(ids)) for ids in encoded
            ]

        if return_tensors == "pt":
            return {"input_ids": torch.tensor(encoded, dtype=torch.long)}

        return {"input_ids": encoded}


class _Frame2seqLogitFormatter(LogitFormatter):
    """Block unknown and mask outputs while preserving dense logits elsewhere."""

    def __init__(self, unk_token_id: int, mask_token_id: int):
        self.unk_token_id = unk_token_id
        self.mask_token_id = mask_token_id

    def __call__(
        self, logits: torch.Tensor, input_ids: torch.LongTensor
    ) -> torch.FloatTensor:
        logits_SPT = logits.float().clone()  # logits_SPT [S, P, T] - cloned logits
        logits_SPT[..., self.unk_token_id] = float("-inf")
        logits_SPT[..., self.mask_token_id] = float("-inf")
        return logits_SPT


class _Frame2seqEnsemble(nn.Module):
    def __init__(self, models: list[_Frame2seqModel]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(
        self,
        X_BPA3: torch.FloatTensor,
        seq_mask_BP: torch.BoolTensor,
        input_ohe_BPU: torch.FloatTensor,
    ) -> torch.FloatTensor:
        logits_sum_BPU = torch.zeros(
            X_BPA3.shape[0],
            X_BPA3.shape[1],
            21,
            device=X_BPA3.device,
            dtype=X_BPA3.dtype,
        )  # logits_sum_BPU [B, P, U] - running sum over ensemble member logits
        for model in self.models:
            logits_member_BPU = model(
                X_BPA3, seq_mask_BP, input_ohe_BPU
            )  # logits_member_BPU [B, P, U] - one model's 21-token logits
            logits_sum_BPU = (
                logits_sum_BPU + logits_member_BPU
            )  # logits_sum_BPU [B, P, U]
        return logits_sum_BPU / len(self.models)  # logits_mean_BPU [B, P, U]


class Frame2seq(GenerativeModelWithEmbedding):
    """Frame2seq structure-conditioned inverse folding model.

    Loads Frame2seq's bundled checkpoint ensemble and exposes it through
    protstar's GenerativeModelWithEmbedding interface.

    Conditioning is required and must be set with:

    ``model.set_condition_({"pdb_path": "1abc.pdb", "chain_id": "A"})``

    Tensor Index Legend:
        B: batch index
        P: residue position index
        A: atom index (N, CA, C, CB, O)
        U: Frame2seq sequence dim (21 = 20 AAs + X)
        T: protstar tokenizer dim (22 = U + <mask>)
        D: single-model embedding dim (128)
        E: concatenated ensemble embedding dim (D * n_models)
    """

    OUTPUT_DIM = 22
    SEQUENCE_DIM = 21

    def __init__(self, checkpoint_paths: list[str | Path] | None = None):
        self._checkpoint_paths = [
            str(path) for path in self._resolve_checkpoint_paths(checkpoint_paths)
        ]

        models = [
            _Frame2seqModel.load_from_checkpoint(path, map_location="cpu").eval()
            for path in self._checkpoint_paths
        ]
        first_model = models[0]

        for model in models[1:]:
            assert model.single_dim == first_model.single_dim
            assert model.sequence_dim == first_model.sequence_dim

        self._n_models = len(models)
        self._single_emb_dim = first_model.single_dim
        self.EMB_DIM = self._single_emb_dim * self._n_models

        tokenizer = Frame2seqTokenizer(include_mask_token=True)
        unk_token_id = tokenizer.vocab["X"]
        assert tokenizer.mask_token_id is not None
        mask_token_id = tokenizer.mask_token_id
        logit_formatter = _Frame2seqLogitFormatter(unk_token_id, mask_token_id)

        super().__init__(
            model=_Frame2seqEnsemble(models),
            tokenizer=tokenizer,
            logit_formatter=logit_formatter,
        )

    @staticmethod
    def _resolve_checkpoint_paths(
        checkpoint_paths: list[str | Path] | None,
    ) -> list[Path]:
        if checkpoint_paths is not None:
            resolved_paths = [Path(path) for path in checkpoint_paths]
        else:
            package_dir = Path(_frame2seq_init_file).resolve().parent
            checkpoint_dir = package_dir / "trained_models"
            resolved_paths = sorted(checkpoint_dir.glob("*.ckpt"))

        if len(resolved_paths) == 0:
            raise ValueError("No Frame2seq checkpoint files were found.")

        for path in resolved_paths:
            if not path.exists():
                raise ValueError(f"Frame2seq checkpoint does not exist: {path}")

        return resolved_paths

    @staticmethod
    def condition_from_pdb(
        pdb_path: str | Path, chain_id: str
    ) -> Frame2seqConditioning:
        return {"pdb_path": str(pdb_path), "chain_id": chain_id}

    def _save_args(self) -> dict:
        return {"checkpoint_paths": self._checkpoint_paths}

    def preprocess_observations(
        self, observations: Frame2seqConditioning
    ) -> _Frame2seqCachedConditioning:
        seq_mask_BP, native_seq_BP, X_BPA3 = get_inference_inputs(
            str(observations["pdb_path"]), observations["chain_id"]
        )

        X_LA3 = X_BPA3.squeeze(0)  # X_LA3 [P, A, 3] - chain atom coordinates
        seq_mask_L = seq_mask_BP.squeeze(
            0
        ).bool()  # seq_mask_L [P] - valid residue mask
        native_seq_L = native_seq_BP.squeeze(
            0
        ).long()  # native_seq_L [P] - native Frame2seq token ids

        return {
            "X_LA3": X_LA3,
            "seq_mask_L": seq_mask_L,
            "native_seq_L": native_seq_L,
        }

    def _to_frame2seq_vocab(self, ohe_seq_BPT: torch.FloatTensor) -> torch.FloatTensor:
        input_ohe_BPU = ohe_seq_BPT[
            :, :, : self.SEQUENCE_DIM
        ].clone()  # input_ohe_BPU [B, P, U] - AA+X channels
        extra_mass_BP = ohe_seq_BPT[:, :, self.SEQUENCE_DIM :].sum(
            dim=-1
        )  # extra_mass_BP [B, P] - total probability on extra channels
        input_ohe_BPU[:, :, 20] = (
            input_ohe_BPU[:, :, 20] + extra_mass_BP
        )  # input_ohe_BPU [B, P, U] - fold extra mass into unknown token
        return input_ohe_BPU

    def _single_model_embedding(
        self,
        model: _Frame2seqModel,
        X_BPA3: torch.FloatTensor,
        seq_mask_BP: torch.BoolTensor,
        input_ohe_BPU: torch.FloatTensor,
    ) -> torch.FloatTensor:
        rigid_BPR = Rigid.from_3_points(
            X_BPA3[:, :, 0], X_BPA3[:, :, 1], X_BPA3[:, :, 2]
        )  # rigid_BPR [B, P] - residue rigid frames from backbone atoms

        s_init_BPF, in_s_BPD = make_s_init(
            model, X_BPA3, input_ohe_BPU, seq_mask_BP
        )  # s_init_BPF [B, P, single_dim+6], in_s_BPD [B, P, D]
        s_BPD = model.sequence_to_single(
            s_init_BPF
        )  # s_BPD [B, P, D] - geometry+position to single representation
        s_BPD = s_BPD + model.input_sequence_layer_norm(
            in_s_BPD
        )  # s_BPD [B, P, D] - add masked input-sequence embedding

        z_init_BPPF = make_z_init(
            model, X_BPA3
        )  # z_init_BPPF [B, P, P, pair_dim] - initial pairwise features
        z_BPPD = model.edge_to_pair(
            z_init_BPPF
        )  # z_BPPD [B, P, P, D] - pair representation

        seq_mask_long_BP = (
            seq_mask_BP.long()
        )  # seq_mask_long_BP [B, P] - IPA mask as int
        attn_drop_rate = 0.2 if model.training else 0.0

        for layer in model.layers:
            ipa, ipa_dropout, layer_norm_ipa, *transit_layers, edge_transition = layer

            ipa_update_BPD = ipa(
                s_BPD,
                z_BPPD,
                rigid_BPR,
                seq_mask_long_BP,
                attn_drop_rate=attn_drop_rate,
            )  # ipa_update_BPD [B, P, D] - IPA single-state update
            s_BPD = s_BPD + ipa_update_BPD  # s_BPD [B, P, D]
            s_BPD = ipa_dropout(s_BPD)  # s_BPD [B, P, D]
            s_BPD = layer_norm_ipa(s_BPD)  # s_BPD [B, P, D]

            if model.st_mod_tsit_factor > 1:
                pre_transit = transit_layers[0]
                transition = transit_layers[1]
                post_transit = transit_layers[2]

                s_BPD = pre_transit(
                    s_BPD
                )  # s_BPD [B, P, D*factor] - transition expansion
                s_BPD = transition(s_BPD)  # s_BPD [B, P, D*factor] - transition block
                s_BPD = post_transit(s_BPD)  # s_BPD [B, P, D] - transition projection
            else:
                transition = transit_layers[0]
                s_BPD = transition(s_BPD)  # s_BPD [B, P, D] - transition block

            if edge_transition is not None:
                z_BPPD = edge_transition(
                    s_BPD, z_BPPD
                )  # z_BPPD [B, P, P, D] - pair update conditioned on singles

        return s_BPD

    def _differentiable_embedding_with_observations(
        self,
        ohe_seq_BPT: torch.FloatTensor,
        observations: _Frame2seqCachedConditioning | dict[str, torch.Tensor],
    ) -> torch.FloatTensor:
        B, P, _ = ohe_seq_BPT.shape
        device = ohe_seq_BPT.device

        input_ohe_BPU = self._to_frame2seq_vocab(
            ohe_seq_BPT
        )  # input_ohe_BPU [B, P, U] - Frame2seq AA+X input distribution

        if observations["X_LA3"].dim() == 3:
            dummy_batch_B = torch.zeros(
                B, dtype=torch.long, device=device
            )  # dummy_batch_B [B] - batch-size carrier for default collator
            obs = self.collate_observations(dummy_batch_B, observations)
        else:
            obs = observations

        X_BPA3 = obs["X_LA3"].to(
            device=device
        )  # X_BPA3 [B, P, A, 3] - conditioned coordinates
        seq_mask_BP = obs["seq_mask_L"].to(
            device=device, dtype=torch.bool
        )  # seq_mask_BP [B, P] - conditioned valid-position mask

        assert X_BPA3.shape[:2] == (B, P)
        assert seq_mask_BP.shape == (B, P)

        embeddings_by_model = []
        for model in self.model.models:
            emb_BPD = self._single_model_embedding(
                model=model,
                X_BPA3=X_BPA3,
                seq_mask_BP=seq_mask_BP,
                input_ohe_BPU=input_ohe_BPU,
            )  # emb_BPD [B, P, D] - one ensemble member single representation
            embeddings_by_model.append(emb_BPD)

        emb_BPE = torch.cat(
            embeddings_by_model, dim=-1
        )  # emb_BPE [B, P, E] - concatenated ensemble embeddings
        return emb_BPE

    def differentiable_embedding(
        self, ohe_seq_SPT: torch.FloatTensor
    ) -> torch.FloatTensor:
        if self.observations is None:
            raise ValueError(
                "Frame2seq requires structure conditioning. "
                "Call set_condition_() or use conditioned_on() first."
            )
        return self._differentiable_embedding_with_observations(
            ohe_seq_BPT=ohe_seq_SPT,
            observations=self.observations,
        )

    def forward(self, seq_SP: torch.LongTensor, **kwargs) -> torch.FloatTensor:
        ohe_BPT = F.one_hot(seq_SP, num_classes=self.tokenizer.vocab_size).float()
        if kwargs:
            emb_BPE = self._differentiable_embedding_with_observations(
                ohe_seq_BPT=ohe_BPT,
                observations=kwargs,
            )
        else:
            emb_BPE = self.differentiable_embedding(ohe_BPT)
        return self.embedding_to_outputs(emb_BPE)

    @staticmethod
    def _pad_logits(logits_BPU: torch.FloatTensor) -> torch.FloatTensor:
        pad_mask_BP1 = torch.full(
            (*logits_BPU.shape[:-1], 1),
            float("-inf"),
            device=logits_BPU.device,
            dtype=logits_BPU.dtype,
        )  # pad_mask_BP1 [B, P, 1] - mask-token output column
        logits_BPT = torch.cat(
            [logits_BPU, pad_mask_BP1], dim=-1
        )  # logits_BPT [B, P, T] - logits with extra mask-token column
        return logits_BPT

    def embedding_to_outputs(
        self, embedding_SPD: torch.FloatTensor
    ) -> torch.FloatTensor:
        embedding_chunks = torch.split(
            embedding_SPD, self._single_emb_dim, dim=-1
        )  # embedding_chunks list[[B, P, D]] - one chunk per ensemble member
        assert len(embedding_chunks) == self._n_models

        logits_sum_BPU = torch.zeros(
            embedding_SPD.shape[0],
            embedding_SPD.shape[1],
            self.SEQUENCE_DIM,
            device=embedding_SPD.device,
            dtype=embedding_SPD.dtype,
        )  # logits_sum_BPU [B, P, U] - running sum over ensemble logits

        for model, emb_chunk_BPD in zip(self.model.models, embedding_chunks):
            logits_member_BPU = model.single_to_sequence(
                emb_chunk_BPD
            )  # logits_member_BPU [B, P, U] - one model's sequence logits
            logits_sum_BPU = (
                logits_sum_BPU + logits_member_BPU
            )  # logits_sum_BPU [B, P, U]

        logits_mean_BPU = logits_sum_BPU / self._n_models  # logits_mean_BPU [B, P, U]
        return self._pad_logits(logits_mean_BPU)

    def format_raw_to_logits(
        self,
        raw_output: torch.FloatTensor,
        seq_SP: torch.LongTensor,
        **kwargs,
    ) -> torch.FloatTensor:
        return self.logit_formatter(raw_output.float(), seq_SP)
