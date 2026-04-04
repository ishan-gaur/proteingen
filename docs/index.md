## What is ProteinGen?

ProteinGen makes it easy to use cutting-edge machine learning methods for protein engineering. ProteinGen provides:





1. the latest workflows for leveraging your wet-lab data to design new libraries, and


2. all the common protein sequence models (incl. inverse-folding), reimplemented to work with our workflows out-of-the-box.








Our framework introduces new abstractions for defining generative and predictive models that ensure interoperability between various training, sampling, and scoring strategies. It *code*-ifies the insights from our recent [theoretical unification](https://arxiv.org/abs/2505.04823) of this area and has drastically reduced the work for us to develop new machine learning methods in our own lab. We use ProteinGen when working with [our collaborators](http://www.jennifer.listgarten.com/group.html#:~:text=Collaborators) as well.




For our computational colleagues, we hope the ProteinGen codebase makes your lives easier, and for our wet-lab counterparts, we hope this makes the latest ML techniques more accessible to you. Let's engineer some amazing new proteins together!




<!-- Without ProteinGen, every time you try a different model, training, or sampling algorithm, you basically have to rewrite your existing code from scratch. We dealt with this ourselves when developing new methods in this area. Every experiment and every paper replicated weeks of work. We wanted to hide away in our code complexity that you shouldn't have to know, like that ESM only uses 33 of its 64 output logits, and prevent you from having to read complicated framework code, like the 1500 line ProteinMPNN [input datastructure](https://github.com/RosettaCommons/foundry/blob/c5a9fbefdeb2c9b107c347aee693d5166d73fa70/models/mpnn/src/mpnn/utils/inference.py#L678) -->





<!--TODO[pi] We should put this line somewhere in the conceptual overview of the design; Pipelines written with ProteinGen make it trivial to swap in and out models, training techniques, and inference time algorithms whenever you want. Implementation costs should never stop you from trying the latest and greatest technique for protein design.-->




ProteinGen was developed by [Ishan Gaur](https://ishangaur.com) and is maintained by the [Listgarten Lab](http://www.jennifer.listgarten.com/group.html) at UC Berkeley.

### Switching Models and Algorithms Made Easy


Let's take the stability optimization experiment from [ProteinGuide](https://arxiv.org/abs/2505.04823) as an example. The paper presents two guidance algorithms: TAG (fast, gradient-based) and DEG (exact, enumeration-based). The original paper used TAG with PMPNN, but ProteinGen makes it super easy to use DEG instead or switch to using ESM3:


=== "TAG + PMPNN"

    ```python hl_lines="2 3 12 16"
    from proteingen.models import PMPNN, StabilityPredictor
    from proteingen.guide import TAG
    from proteingen.sampling import sample_euler

    coords = ... # load backbone structure from pdb file

    # Load the models
    gen_model = PMPNN().conditioned_on({"structure": coords}) # inverse-folding model
    predictor = StabilityPredictor() # ddg predictor trained on the Megascale dataset

    # Get the stability guided conditional generative model
    tag = TAG(gen_model, predictor).cuda()

    # Sample 8 stability-optimized variants starting from fully masked sequences
    masked_seqs = ["<mask>" * 100] * 8
    seqs = sample_euler_integration(tag, masked_seqs)
    ```

=== "DEG + PMPNN"

    ```python hl_lines="2 3 12 16"
    from proteingen.models import PMPNN, StabilityPredictor
    from proteingen.guide import DEG
    from proteingen.sampling import sample_ancestral

    coords = ... # load backbone structure from pdb file

    # Load the models
    gen_model = PMPNN().conditioned_on({"structure": coords}) # inverse-folding model
    predictor = StabilityPredictor() # ddg predictor trained on the Megascale dataset

    # Get the stability guided models
    tag = DEG(gen_model, predictor).cuda()

    # Sample 8 stability-optimized variants starting from fully masked sequences
    masked_seqs = ["<mask>" * 100] * 8
    seqs = sample_ancestral(tag, masked_seqs)
    ```

=== "DEG + ESM3"

    ```python hl_lines="1 8"
    from proteingen.models import ESM3, StabilityPredictor
    from proteingen.guide import DEG
    from proteingen.sampling import sample_ancestral

    coords = ... # load backbone structure from pdb file

    # Load the models
    gen_model = ESM3("esm3-small").conditioned_on({"structure": coords}) # inverse-folding model
    predictor = StabilityPredictor() # ddg predictor trained on the Megascale dataset

    # Get the stability guided models
    tag = DEG(gen_model, predictor).cuda()

    # Sample 8 stability-optimized variants starting from fully masked sequences
    masked_seqs = ["<mask>" * 100] * 8
    seqs = sample_ancestral(tag, masked_seqs)
    ```




### Built with Agents in Mind


We're excited about the future of AI coding agents but, as scientists, we recognize that it's tricky to make sure you trust their results.


Our documentation and codebase are structured to empower you to leverage agents as safely as possible when using ProteinGen. The [Workflows](workflows/index.md) section includes guides to help you understand the algorithms in our repo and give you a checklist of evaluations to run at each step to check the quality of the agents' work. These evaluation checklists are the same ones we use with our collaborators. They are being continuously updated as we learn more through our own work. Also make sure to follow the [Setup](setup/index.md) instructions so that your agents get our AGENTS.md and SKILLS.md files, and avoid the common mistakes we uncovered during our own testing.



### Are We Missing a Model?

No problem! We've included detailed workflows for adding new models in our [Contributing](contributing/index.md) section. In there, you'll find a robust agent workflow that we've included as a SKILL.md file in the repo. It was able to autonomously integrate and document ProteinMPNN into the codebase with minimal intervention on our part. All you have to do is ask:

> "Read the skill file at `.agents/skills/add-generative-model/SKILL.md` and follow it to add **[model name]** to ProteinGen."



### Share Your Work on ProteinGen


We want to make it easy for people to use cutting edge models and design algorithms. A key part of this is making it easy for you to get your own work out there. In our [Contributing](contributing.md) section, you can find instructions to submit new models or sampling algorithms to be included in the next release of ProteinGen. We've also created specialized SKILL.md files that walk your coding agents through the process of adding, testing, and submitting pull requests add your codebase. We'd love to include your work, even if you've never contributed to open source before!


<!-- TODO[pi]: flesh out home page with a diagram showing the generative + predictive model combination via Bayes' rule -->
<!-- TODO[pi]: add a quick "5-line example" code block showing unconditional sampling -->
