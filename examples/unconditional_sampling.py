from proteingen.models import ESMC
from proteingen import sample_any_order

n_samples = 5
len_sample = 256

model = ESMC().cuda()
initial_x = ["<mask>" * len_sample for _ in range(n_samples)]

gen_sample_seqs = sample_any_order(model, initial_x)
print(gen_sample_seqs)
