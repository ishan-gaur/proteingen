from proteingen.models import ESMC
from proteingen import sample

n_samples = 5
len_sample = 256

model = ESMC().cuda()
initial_x = ["<mask>" * len_sample for _ in range(n_samples)]

gen_sample_seqs = sample(model, initial_x)["sequences"]
print(gen_sample_seqs)
