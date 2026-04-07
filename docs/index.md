New here? Start with: [**Setup**](setup.md) · [**Examples**](examples/index.md) · [**Workflows**](workflows/index.md) · [**Design Philosophy**](reference/design-philosophy.md)

---

## What is ProteinGen?

Today, every time you need to try a different model, training, or sampling algorithm, you basically have to rewrite your existing code from scratch. Even worse to use the latest sampling and design techniques you might have to wade through hundreds of lines of library code to figure out, for example, that ESM only uses 33 of their 64 output logits or exactly how you were supposed to format conditioning inputs for ProteinMPNN. 

Pipelines written with ProteinGen make it trivial to swap in and out models, training techniques, and inference time algorithms whenever you want. Implementation costs should never stop you from trying the latest and greatest technique for protein design.

ProteinGen centralizes common models and computational workflows for sequence-based protein design in one place. We provide a unified interface that makes models from different organizations--ESM, Evodiff, DPLM, ProteinMPNN--interoperate. This lets us implement training, sampling, and guidance algorithms on top of them as part of a single ecosystem. Our goal is to accelerate the development of new methods for drylab practitioners and to make modern statistical and deep learning methods for protein design more accessible for the wetlab community as well.

ProteinGen was developed by [Ishan Gaur](https://ishangaur.com) and is maintained by the [Listgarten Lab](http://www.jennifer.listgarten.com/group.html) at UC Berkeley.

### Trying new models is dead simple

For example, this is all the machine learning code you need to replicate the stability optimization experiments from our recent [ProteinGuide preprint](https://arxiv.org/abs/2505.04823). In that paper we present two guidance algorithms: TAG (fast, gradient-based) and DEG (exact, enumeration-based). With ProteinGen, switching between algorithms or swapping in a completely different generative model is as easy as changing the imports:

=== "TAG + ESM3"

    ```python hl_lines="2 3 11 15"
    from proteingen.models import ESM3, StabilityPredictor
    from proteingen.guide import TAG
    from proteingen.sampling import sample_euler

    # Load the models
    coords = ... # load backbone structure from pdb file
    gen_model = ESM3("esm3-small").conditioned_on({"structure": coords})
    predictor = StabilityPredictor()

    # Get the stability guided models
    tag = TAG(gen_model, predictor).cuda()

    # Have it generate 8 stability-optimized sequences of length 100.
    masked_seqs = ["<mask>" * 100] * 8
    seqs = sample_euler_integration(tag, masked_seqs)
    ```

=== "DEG + ESM3"

    ```python hl_lines="2 3 11 15"
    from proteingen.models import ESM3, StabilityPredictor
    from proteingen.guide import DEG
    from proteingen.sampling import sample_ancestral

    # Load the models
    coords = ... # load backbone structure from pdb file
    gen_model = ESM3("esm3-small").conditioned_on({"structure": coords})
    predictor = StabilityPredictor()

    # Get the stability guided models
    tag = DEG(gen_model, predictor).cuda()

    # Have it generate 8 stability-optimized sequences of length 100.
    masked_seqs = ["<mask>" * 100] * 8
    seqs = sample_ancestral(tag, masked_seqs)
    ```

=== "DEG + PMPNN"

    ```python hl_lines="1 7"
    from proteingen.models import PMPNN, StabilityPredictor
    from proteingen.guide import DEG
    from proteingen.sampling import sample_ancestral

    # Load the models
    coords = ... # load backbone structure from pdb file
    gen_model = PMPNN.conditioned_on({"structure": coords})
    predictor = StabilityPredictor()

    # Get the stability guided models
    tag = DEG(gen_model, predictor).cuda()

    # Have it generate 8 stability-optimized sequences of length 100.
    masked_seqs = ["<mask>" * 100] * 8
    seqs = sample_ancestral(tag, masked_seqs)
    ```

### Use coding agents with confidence

As part of minimizing implementation overhead, we're obviously excited about the future of AI coding agents but recognize that it can be hard to balance their use with doing science that you trust. 

We have structured our codebase and documentation to empower you to develop new design pipelines as effectively as possible. We provide resources for you to understand the algorithms in our repo, make sure your agents avoid common mistakes, and give you the evaluation tools to verify the quality of the pipelines your agents create. Specifically, when developing this repo we focused on the following three things:

1. [Workflows](workflows/index.md): these guides walk you through common training, sampling, and conditional generation pipelines step-by-step. They provide conceptual overviews, agent prompts, and the evals/sanity checks we run when writing code ourselves to make sure everything looks good in our own wetlab collaborations.

2. We teach your coding agents how to use our code. Having Claude Code read through our repo shouldn't torch your token budget. We include AGENTS.md files and SKILLS.md files to help your models understand our design philosophy, common gotchas, and best practices without you having to manually intervene.

3. [Contributing](contributing.md): we want to make it easy for people to use cutting edge models and design algorithms. A key part of this is making it easy for you to get your own work out there. We include SKILLS.md files that help your coding agents make your research code ProteinGen compatible and submit pull requests to add your work to the next release. We want to make this possible even if you've never contributed to open source before.

<!-- TODO[pi]: flesh out home page with a diagram showing the generative + predictive model combination via Bayes' rule -->
<!-- TODO[pi]: add a quick "5-line example" code block showing unconditional sampling -->
