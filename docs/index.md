## What is ProteinGen?

ProteinGen makes it easy to use cutting-edge machine learning methods for protein engineering. It provides:

1. the latest workflows for leveraging your wet-lab data to design new libraries, and
2. all the common protein sequence models (incl. inverse-folding), reimplemented to work with our workflows out-of-the-box.

Our framework *code*-ifies the insights from our recent [theoretical unification](https://arxiv.org/abs/2505.04823) of generative and predictive protein models, ensuring interoperability between various training, sampling, and scoring strategies. It has drastically reduced the work to develop new methods in our own research, and we use it with our [wet-lab collaborators](http://www.jennifer.listgarten.com/group.html#:~:text=Collaborators) as well.

For our computational colleagues, we hope ProteinGen makes your lives easier. For our wet-lab counterparts, we hope it makes the latest ML techniques more accessible. Let's engineer some amazing new proteins together!

<!-- Without ProteinGen, every time you try a different model, training, or sampling algorithm, you basically have to rewrite your existing code from scratch. We dealt with this ourselves when developing new methods in this area. Every experiment and every paper replicated weeks of work. We wanted to hide away in our code complexity that you shouldn't have to know, like that ESM only uses 33 of its 64 output logits, and prevent you from having to read complicated framework code, like the 1500 line ProteinMPNN [input datastructure](https://github.com/RosettaCommons/foundry/blob/c5a9fbefdeb2c9b107c347aee693d5166d73fa70/models/mpnn/src/mpnn/utils/inference.py#L678) -->

<!--TODO[pi] We should put this line somewhere in the conceptual overview of the design; Pipelines written with ProteinGen make it trivial to swap in and out models, training techniques, and inference time algorithms whenever you want. Implementation costs should never stop you from trying the latest and greatest technique for protein design.-->

ProteinGen was developed by [Ishan Gaur](https://ishangaur.com) and is maintained by the [Listgarten Lab](http://www.jennifer.listgarten.com/group.html) at UC Berkeley.

```python
from proteingen.models.mpnn import ProteinMPNN
from proteingen.models.utils import load_pdb
from proteingen.sampling import sample_any_order_ancestral

structure = load_pdb("1YCR.pdb")
model = ProteinMPNN().conditioned_on({"structure": structure})

seqs = sample_any_order_ancestral(model, ["<mask>" * 98] * 8)
```

### Switching Models and Algorithms Made Easy

Take the stability optimization experiment from [ProteinGuide](https://arxiv.org/abs/2505.04823) as an example. The paper presents two guidance algorithms — TAG (gradient-based) and DEG (enumeration-based) — and originally used TAG with PMPNN. With ProteinGen, switching to DEG or swapping in ESM3 is just a change of imports:

=== "TAG + PMPNN"

    ```python hl_lines="2 3 12 16"
    from proteingen.models import PMPNN, StabilityPredictor
    from proteingen.guide import TAG
    from proteingen.sampling import sample_linear_interpolation

    coords = ... # load backbone structure from pdb file

    # Load the models
    gen_model = PMPNN().conditioned_on({"structure": coords}) # inverse-folding model
    predictor = StabilityPredictor() # ddg predictor trained on the Megascale dataset

    # Get the stability guided conditional generative model
    guided = TAG(gen_model, predictor).cuda()

    # Sample 8 stability-optimized variants starting from fully masked sequences
    masked_seqs = ["<mask>" * 100] * 8
    seqs = sample_linear_interpolation(guided, masked_seqs)
    ```

=== "DEG + PMPNN"

    ```python hl_lines="2 3 12 16"
    from proteingen.models import PMPNN, StabilityPredictor
    from proteingen.guide import DEG
    from proteingen.sampling import sample_any_order_ancestral

    coords = ... # load backbone structure from pdb file

    # Load the models
    gen_model = PMPNN().conditioned_on({"structure": coords}) # inverse-folding model
    predictor = StabilityPredictor() # ddg predictor trained on the Megascale dataset

    # Get the stability guided models
    guided = DEG(gen_model, predictor).cuda()

    # Sample 8 stability-optimized variants starting from fully masked sequences
    masked_seqs = ["<mask>" * 100] * 8
    seqs = sample_any_order_ancestral(guided, masked_seqs)
    ```

=== "DEG + ESM3"

    ```python hl_lines="1 8"
    from proteingen.models import ESM3, StabilityPredictor
    from proteingen.guide import DEG
    from proteingen.sampling import sample_any_order_ancestral

    coords = ... # load backbone structure from pdb file

    # Load the models
    gen_model = ESM3("esm3-small").conditioned_on({"structure": coords}) # inverse-folding model
    predictor = StabilityPredictor() # ddg predictor trained on the Megascale dataset

    # Get the stability guided models
    guided = DEG(gen_model, predictor).cuda()

    # Sample 8 stability-optimized variants starting from fully masked sequences
    masked_seqs = ["<mask>" * 100] * 8
    seqs = sample_any_order_ancestral(guided, masked_seqs)
    ```

### Built with Agents in Mind

We're excited about AI coding agents but, as scientists, recognize it's tricky to trust their results. Our [Workflows](workflows/index.md) include algorithm guides and evaluation checklists at each step — the same ones we use with our collaborators, continuously updated as we learn more. Follow the [Setup](setup/index.md) instructions to give your agents our AGENTS.md and SKILLS.md files so they avoid common mistakes we uncovered during testing.

### Share Your Work on ProteinGen

We want to make it easy for you to get your work out there. Our [Contributing](contributing/index.md) section has instructions for submitting new models or sampling algorithms to be included in the next release. We've also created SKILL.md files that walk your coding agents through the process. We'd love to include your work, even if you've never contributed to open source before!

### Are We Missing a Model?

No problem. Our Contributing section includes an agent workflow that autonomously integrated ProteinMPNN into the codebase with minimal intervention on our part. If there's a model you want, all you have to do is ask:

> "Read the skill file at `.agents/skills/add-generative-model/SKILL.md` and follow it to add **[model name]** to ProteinGen."

<!-- TODO[pi]: flesh out home page with a diagram showing the generative + predictive model combination via Bayes' rule -->
<!-- TODO[pi]: add a quick "5-line example" code block showing unconditional sampling -->
