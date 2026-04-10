from protstar.models import ESM3, ExprStabilityProbePredictor
from protstar.datasets.seki_tyrosine_kinase import WT_LABELS
import atomworks.io as aio

# In this script, we use ESM3 unconditionally and with a pretrained predictor
# to estimate the boost in spearman correlation achieved by using a conditional model
test_set_SP = None  # TODO[pi] test set tensordataset tensor
# TODO[pi] need to find where I implemented the stuff to get the likelihood trajectories--maybe in the tanja repo? actually probably in esm-cath
tkdomain_structure = aio.parse(ExprStabilityProbePredictor.TK_DOMAIN_PDB_PATH)
model = ESM3().set_condition({"coords_RAX": tkdomain_structure})
pred_model = ExprStabilityProbPredictor()
y = [(">", WT_LABELS[0]), (">", WT_LABELS[1])]

uncond_mean_LL_SW = path_likelihood(model, points=10) # TODO[pi]: W is the decoding path index
with pred_model.with_target(y):
