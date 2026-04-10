from protstar.models import ESMC
from protstar.sampling import sample

len_sample = 128
n_samples = 8
masked_seqs = ["<mask>" * len_sample] * n_samples

model = ESMC("esmc_300m").cuda()
seqs = sample(model, masked_seqs)["sequences"]  # 8 random proteins
