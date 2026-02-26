from dfm.models import ESMC
from dfm import sample_any_order_ancestral

n_samples = 5
len_sample = 256

model = ESMC().cuda()
initial_x = ["<mask>" * len_sample for _ in range(n_samples)]

gen_sample_seqs = sample_any_order_ancestral(model, initial_x)
print(gen_sample_seqs)
