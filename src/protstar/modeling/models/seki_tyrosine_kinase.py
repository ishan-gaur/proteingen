from protstar.modeling import ESM3
from protstar.modeling import PredictiveModel, LinearProbe
import atomworks.io as aio
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer


class ExprStabilityProbePredictor(RealValuedEnsembleModel):
    """
    Expression and Stability predictor for EphB1 receptor tyrosine kinase domain (UniProt P54762)
    Trained on data from Seki et al. 2025 https://www.biorxiv.org/content/10.1101/2025.08.03.668353v1
    The dataset tests PMPNN and
    """

    TK_DOMAIN_PDB_PATH = None  # TODO[pi] download the structure and figure out how we should store this sort of data, maybe take the directory to put this in as an argument? In the huggingface repo??

    def __init__(self):
        tk_domain_structure = aio.parse(ExprStabilityProbePredictor.TK_DOMAIN_PDB_PATH)
        self.probe = LinearProbe(
            embed_ESM3.set_condition({"structure": tk_domain_structure}), 2
        )
        self.tokenizer = EsmSequenceTokenizer()
        self.input_dim = (
            self.probe.embed_model.tokenizer.vocab_size
        )  # make this the number of rows in the embedding module

    # TODO[pi]
    def load_hf_checkpoint():
        raise NotImplementedError()

    def forward(self, ohe_seq_SP):
        return self.probe.forward(ohe_seq_SP)
