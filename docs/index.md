ProteinGen is a package for library design with machine learning. It focuses on leveraging assay-labeled data to improve libraries sampled using protein sequence models.


In order to make writing library design [pipelines](workflows/index.md) easier, we created a simplified [interface](reference/design-philosophy.md#design-philosophy) for using sequence models. Below is an example of inverse-folding with ProteinMPNN using ProteinGen. On the other tab, you can see the *forty-five lines* needed for the original codebase.





We similarly provide simplified APIs to a broad array of protein [models](models/index.md) including ESM3, DPLM2, and ProGen3.

















=== "ProteinGen Inverse-Folding"

    ```python
    from proteingen.models.mpnn import ProteinMPNN
    from proteingen.models.utils import load_pdb
    from proteingen.sampling import sample

    structure = load_pdb("1YCR.pdb")
    masked_seqs = ["<mask>" * 98] * 8 # placeholders to be designed

    model = ProteinMPNN().conditioned_on({"structure": structure}) # configure inverse-folding
    seqs = sample(model, masked_seqs)["sequences"] # generate sequences
    ```

=== "Original ProteinMPNN"

    ```python
    import copy, torch, numpy as np
    from protein_mpnn_utils import (
        parse_PDB, StructureDatasetPDB, ProteinMPNN,
        tied_featurize, _S_to_seq,
    )

    # Step 1: Parse PDB and build dataset
    pdb_dict_list = parse_PDB("1YCR.pdb", ca_only=False)
    dataset = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=200000)

    # Step 2: Build chain design specification
    all_chains = [k[-1:] for k in pdb_dict_list[0] if k[:9] == "seq_chain"]
    chain_id_dict = {pdb_dict_list[0]["name"]: (all_chains, [])}

    # Step 3: Load model with architecture params from checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load("vanilla_model_weights/v_48_020.pt", map_location=device)
    model = ProteinMPNN(
        ca_only=False, num_letters=21, node_features=128, edge_features=128,
        hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
        augment_eps=0.0, k_neighbors=ckpt["num_edges"],
    )
    model.to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Step 4: Featurize — returns 20 tensors
    batch_clones = [copy.deepcopy(dataset[0]) for _ in range(8)]
    (X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list,
     visible_list_list, masked_list_list, masked_chain_length_list_list,
     chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask,
     tied_pos_list_of_lists_list, pssm_coef, pssm_bias,
     pssm_log_odds_all, bias_by_res_all, tied_beta,
    ) = tied_featurize(
        batch_clones, device, chain_id_dict,
        None, None, None, None, None, ca_only=False,
    )

    # Step 5: Sample
    sample_dict = model.sample(
        X, torch.randn(chain_M.shape, device=device), S, chain_M,
        chain_encoding_all, residue_idx, mask=mask, temperature=0.1,
        omit_AAs_np=np.zeros(21), bias_AAs_np=np.zeros(21),
        chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask,
        pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=0.0,
        pssm_log_odds_flag=False, pssm_log_odds_mask=None,
        pssm_bias_flag=False, bias_by_res=bias_by_res_all,
    )

    # Step 6: Decode token indices to sequences
    seqs = [_S_to_seq(sample_dict["S"][i], chain_M[i]) for i in range(8)]
    ```


ProteinGen was developed by [Ishan Gaur](https://ishangaur.com) and is maintained by the [Listgarten Lab](http://www.jennifer.listgarten.com/group.html) at UC Berkeley.







## Why ProteinGen?





ProteinGen makes it easy to use cutting-edge machine learning methods for protein engineering. It provides:

1. the latest workflows for leveraging your wet-lab data to design new libraries, and
2. all the common protein sequence models (incl. inverse-folding), reimplemented to work with our workflows out-of-the-box.

Our framework *code*-ifies the insights from our recent [theoretical unification](https://arxiv.org/abs/2505.04823) of generative and predictive protein models, ensuring interoperability between various training, sampling, and scoring strategies. It has drastically reduced the work to develop new methods in our own research, and we use it with our [wet-lab collaborators](http://www.jennifer.listgarten.com/group.html#:~:text=Collaborators) as well.

<!-- For our computational colleagues, we hope ProteinGen makes your lives easier. For our wet-lab counterparts, we hope it makes the latest ML techniques more accessible. Let's engineer some amazing new proteins together! -->

<!-- Without ProteinGen, every time you try a different model, training, or sampling algorithm, you basically have to rewrite your existing code from scratch. We dealt with this ourselves when developing new methods in this area. Every experiment and every paper replicated weeks of work. We wanted to hide away in our code complexity that you shouldn't have to know, like that ESM only uses 33 of its 64 output logits, and prevent you from having to read complicated framework code, like the 1500 line ProteinMPNN [input datastructure](https://github.com/RosettaCommons/foundry/blob/c5a9fbefdeb2c9b107c347aee693d5166d73fa70/models/mpnn/src/mpnn/utils/inference.py#L678) -->

<!--TODO[pi] We should put this line somewhere in the conceptual overview of the design; Pipelines written with ProteinGen make it trivial to swap in and out models, training techniques, and inference time algorithms whenever you want. Implementation costs should never stop you from trying the latest and greatest technique for protein design.-->





### Switching Models and Algorithms Made Easy

Take the stability optimization experiment from [ProteinGuide](https://arxiv.org/abs/2505.04823) as an example. The paper presents two guidance algorithms — TAG (gradient-based) and DEG (enumeration-based) — and originally used TAG with PMPNN. With ProteinGen, switching to DEG or swapping in ESM3 is just a change of imports:

=== "TAG + PMPNN"

    ```python hl_lines="2 3 13 17"
    from proteingen.models import PMPNN, StabilityPredictor
    from proteingen.guide import TAG
    from proteingen.sampling import sample_ctmc_linear_interpolation
    from proteingen.models.utils import load_pdb

    structure = load_pdb("1YCR.pdb")

    # Load the models
    gen_model = PMPNN().conditioned_on({"structure": structure}) # inverse-folding model
    predictor = StabilityPredictor() # ddg predictor trained on the Megascale dataset

    # Get the stability guided conditional generative model
    guided = TAG(gen_model, predictor).cuda()

    # Sample 8 stability-optimized variants starting from fully masked sequences
    masked_seqs = ["<mask>" * 98] * 8
    seqs = sample_ctmc_linear_interpolation(guided, masked_seqs)
    ```

=== "DEG + PMPNN"

    ```python hl_lines="2 3 13 17"
    from proteingen.models import PMPNN, StabilityPredictor
    from proteingen.guide import DEG
    from proteingen.sampling import sample
    from proteingen.models.utils import load_pdb

    structure = load_pdb("1YCR.pdb")

    # Load the models
    gen_model = PMPNN().conditioned_on({"structure": structure}) # inverse-folding model
    predictor = StabilityPredictor() # ddg predictor trained on the Megascale dataset

    # Get the stability guided models
    guided = DEG(gen_model, predictor).cuda()

    # Sample 8 stability-optimized variants starting from fully masked sequences
    masked_seqs = ["<mask>" * 98] * 8
    seqs = sample(guided, masked_seqs)["sequences"]
    ```

=== "DEG + ESM3"

    ```python hl_lines="1 9"
    from proteingen.models import ESM3, StabilityPredictor
    from proteingen.guide import DEG
    from proteingen.sampling import sample
    from proteingen.models.utils import load_pdb

    structure = load_pdb("1YCR.pdb")

    # Load the models
    gen_model = ESM3("esm3-small").conditioned_on({"structure": structure}) # inverse-folding model
    predictor = StabilityPredictor() # ddg predictor trained on the Megascale dataset

    # Get the stability guided models
    guided = DEG(gen_model, predictor).cuda()

    # Sample 8 stability-optimized variants starting from fully masked sequences
    masked_seqs = ["<mask>" * 98] * 8
    seqs = sample(guided, masked_seqs)["sequences"]
    ```

### Built with Agents in Mind

We're excited about AI coding agents but, as scientists, recognize it's tricky to trust their results. Our [Workflows](workflows/index.md) include algorithm guides and evaluation checklists at each step — the same ones we use with our collaborators, continuously updated as we learn more. Follow the [Setup](setup.md) instructions to give your agents our AGENTS.md and SKILLS.md files so they avoid common mistakes we uncovered during testing.

### Share Your Work on ProteinGen

We want to make it easy for you to get your work out there. Our [Contributing](contributing.md) section has instructions for submitting new models or sampling algorithms to be included in the next release. We've also created SKILL.md files that walk your coding agents through the process. We'd love to include your work, even if you've never contributed to open source before!

## Library Design with ProteinGen

With ProteinGen, designing libraries to optimize some property of a protein involves four modules:

1. **[Data](reference/data.md)** — assay-labeled variants, homologous sequences, noise schedules, and data splits
2. **[Models](reference/models.md)** — generative models (ESM3, ESMC, PMPNN), predictive models (probes, MLPs), guidance (TAG, DEG), and training (LoRA, noisy classifiers)
3. **[Sampling](reference/sampling.md)** — generating a library using the models: discrete-time ancestral, linear interpolation, or flow-matching samplers
4. **[Evaluation](reference/evaluation.md)** — sanity-checking each stage: likelihood curves, oracle agreement, diversity metrics, and structural validation


### Unconditional Sampling (Models + Sampling)

The simplest pipeline: sample from a pretrained model with no data and no property optimization. This demonstrates the **Models** and **Sampling** modules.

```python
from proteingen.models import ESMC
from proteingen.sampling import sample

model = ESMC("esmc_300m").cuda()
seqs = sample(model, ["<mask>" * 100] * 8)["sequences"]  # 8 random proteins
```

That's it — three lines. The model provides `get_log_probs`, the sampler iteratively unmasks positions. See the [unconditional sampling example](examples/unconditional-sampling.md) for details.

### Guided Library Design (All Four Modules)

A realistic pipeline uses all four modules. Here's a sketch of a first-round library design using [ProteinGuide](workflows/protein-guide.md) — combining fine-tuning with classifier guidance to generate variants optimized for a target property.

```python
from proteingen.models import ESMC
from proteingen.models.utils import load_pdb
from proteingen.data import ProteinDataset, uniform_mask_noise, uniform_time
from proteingen.predictive_modeling import OneHotMLP
from proteingen.guide import DEG
from proteingen.sampling import sample

# ── Data ─────────────────────────────────────────────────────────
# Load homologs for fine-tuning and assay-labeled variants for the predictor
homologs = ProteinDataset(sequences=load_homologs("my_protein.fasta"))  # (1)
assay_data = ProteinDataset(
    sequences=labeled_seqs, labels=activity_labels,                     # (2)
)

# ── Models: fine-tune the generative model ───────────────────────
gen_model = ESMC("esmc_300m")
gen_model.apply_lora(r=4)
collate_fn = homologs.collator(gen_model, uniform_mask_noise(gen_model.tokenizer), uniform_time)
# ... training loop (see Fine-tuning workflow) ...                      # (3)

# ── Models: train oracle and noisy predictor ─────────────────────
oracle = MyPredictor(tokenizer=gen_model.tokenizer, ...)                # (4)
# ... train oracle on clean assay data ...

noisy_predictor = MyPredictor(tokenizer=gen_model.tokenizer, ...)       # (5)
# ... train noisy predictor with masked inputs (see ProteinGuide workflow) ...

# ── Evaluation: validate predictor–oracle agreement ──────────────
# Check that the noisy predictor and oracle agree on clean sequences
# before trusting the predictor during sampling                         # (6)
oracle_scores = oracle.predict(val_seqs)
predictor_scores = noisy_predictor.predict(val_seqs)
print(f"Spearman ρ: {spearmanr(oracle_scores, predictor_scores).correlation:.3f}")

# ── Sampling: guided generation ──────────────────────────────────
noisy_predictor.set_target_(True)
noisy_predictor.set_temp_(0.1)          # lower = stronger guidance
guided = DEG(gen_model, noisy_predictor).cuda()

library = sample(guided, ["<mask>" * seq_len] * 100)["sequences"]       # (7)

# ── Evaluation: score the library with the oracle ────────────────
library_scores = oracle.predict(library)                                # (8)
```

1. **Data** — homologous sequences from an MSA for fine-tuning the base model
2. **Data** — assay-labeled variants (e.g. from a DMS or previous round) for training the predictor
3. **Models** — LoRA fine-tuning specializes the base model to your protein family
4. **Models** — the oracle is trained on clean data and used only for evaluation
5. **Models** — the noisy predictor is trained on randomly masked inputs so it works during iterative unmasking
6. **Evaluation** — if the predictor and oracle disagree on clean sequences, the predictor can't be trusted during generation
7. **Sampling** — DEG enumerates all amino acids at each position, reweighting by the predictor's scores
8. **Evaluation** — the oracle scores the final library; these scores inform threshold-setting for the next round

Each numbered annotation maps to a module. The [ProteinGuide workflow](workflows/protein-guide.md) walks through each step in detail, and the [stability-guided generation example](examples/stability-guided-generation.md) shows a working implementation with ESM3 + TAG.
