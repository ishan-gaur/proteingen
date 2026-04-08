# Setup

??? tip "Unfamiliar with the terminal?"
    If you find yourself having trouble with any of the terminal use or package management in the guide below, we recommend checking out MIT's ["Missing Semester of Your CS Education"](https://missing.csail.mit.edu/). The first three videos + the lecture on git and version control should get you up to speed.




## 1. (Recommended) Install a Coding Agent

ProteinGen was designed to facilitate the use of AI coding agents for writing design pipelines.



We recommend [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview) (installation instructions below). There are several other great options, including: [Pi](https://pi.dev), [Amp](https://ampcode.com/install), [Codex](https://developers.openai.com/codex/quickstart). These tools typically require a paid plan with a model provider, but we've gotten plenty of mileage for our tasks out of the basic tiers. 



??? note "Installing Claude Code"

    ```bash
    npm install -g @anthropic-ai/claude-code
    ```

    This requires Node.js ≥ 18. If you don't have Node.js installed:

    === "macOS / Linux (nvm)"

        ```bash
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
        nvm install node
        ```

    === "Homebrew"

        ```bash
        brew install node
        ```

    Then verify:

    ```bash
    claude --version
    ```

    See the [Claude Code quickstart](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview) for authentication setup and first-run instructions.

Once you install the agent, the easiest way to complete the setup is just to copy the link to this guide

```bash
https://ishan-gaur.github.io/proteingen/setup/
```

and ask it to walk you through the setup. If your agent doesn't have internet access, just clone the repo

```bash
git clone https://github.com/ishan-gaur/proteingen.git
```
and point your model to `proteingen/docs/setup.md`. 

If you'd like to continue on manually, we still include the full details to setup ProteinGen below.

## 2. (Optional) Install a Package Manager

We use [uv](https://docs.astral.sh/uv/) for all dependency management and running scripts. [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), [Poetry](https://python-poetry.org/docs/#installation), or good ol' [venv](https://docs.python.org/3/library/venv.html) are also popular options.


!!! tip
    Any `uv pip install ...` command works as plain `pip install ...` from within a conda environment. That being said, *uv is super cool, and we highly recommend giving it a try* — here's a [talk](https://www.youtube.com/watch?v=gSKTfG1GXYQ) from the founder on why package management is an interesting problem.

=== "uv"

    Install uv according to your OS:
    
    === "macOS / Linux"

        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        ```

    === "Windows"

        ```powershell
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```

    === "Homebrew"

        ```bash
        brew install uv
        ```

    See the [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/) for additional options (pip, conda, Docker, etc.).

    After installing, restart your terminal and verify:

    ```bash
    uv --version
    ```

    Create a project:

    ```bash
    mkdir my-protein-project
    cd my-protein-project
    git init
    uv init
    ```

=== "conda"

    We recommend [Miniforge](https://github.com/conda-forge/miniforge#install), which ships with [mamba](https://mamba.readthedocs.io/en/latest/) (a faster drop-in replacement for `conda`):

    === "macOS / Linux"

        ```bash
        curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
        bash Miniforge3-$(uname)-$(uname -m).sh
        ```

    === "Windows"

        Download and run the [Miniforge installer](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe).

    After installing, restart your terminal and verify:

    ```bash
    conda --version
    ```

    Create a project:

    ```bash
    mkdir my-protein-project
    cd my-protein-project
    git init
    conda create -n my-protein-project python=3.12 -y
    conda activate my-protein-project
    ```

## 3. Install ProteinGen

ProteinGen is designed to be a library you own: that you and your agents can adapt as you find what works for you. Accordingly, we recommend doing an editable local install, however you can also install it as a dependency directly from GitHub.

!!! tip "Requirements"
    **Python 3.12** is required (`>=3.12.0, <3.13`). Check with `python --version`. If you're using uv, it will manage the Python version for you. For conda, specify `python=3.12` when creating the environment.
    
    **GPU (CUDA)** is recommended for most models. Smaller models like ProteinMPNN (~1.7M params) run fine on CPU.

=== "Editable Local Install (Recommended)"

    === "uv"

        ```bash
        # Clone ProteinGen alongside your project
        git clone https://github.com/ishan-gaur/proteingen.git
    
        # Add ProteinGen as an editable dependency of your project
        cd my-protein-project
        uv add --editable ../proteingen
        ```

    === "conda / pip"

        ```bash
        # Ensure your Conda environment is active, can skip if using just pip
        conda activate name-of-your-env
    
        # Clone ProteinGen alongside your project
        git clone https://github.com/ishan-gaur/proteingen.git
    
        # Install
        cd proteingen
        pip install -e .
        ```

=== "Direct Install"


    If you don't want to modify ProteinGen's source:

    === "uv"

        ```bash
        uv add "proteingen @ git+https://github.com/ishan-gaur/proteingen.git"
        ```

    === "conda / pip"

        ```bash
        pip install "proteingen @ git+https://github.com/ishan-gaur/proteingen.git"
        ```

### Optional dependencies

Some dependencies are only available via GitHub (not PyPI) and must be installed separately:

**mkdocs-liveedit** — live-reloading plugin for the docs site. Required for `mkdocs serve`:

=== "uv"

    ```bash
    uv pip install "mkdocs-liveedit @ git+https://github.com/ishan-gaur/mkdocs-liveedit.git"
    ```

=== "conda / pip"

    ```bash
    pip install "mkdocs-liveedit @ git+https://github.com/ishan-gaur/mkdocs-liveedit.git"
    ```

**af3-server** — Python client for a persistent AlphaFold 3 inference server. Required for structure prediction workflows (e.g. [Benchmark — Model Families](examples/benchmark-model-families.md), [Fine-tuning Inverse Folding](examples/finetune-inverse-folding.md)):

=== "uv"

    ```bash
    uv pip install "af3-server @ git+https://github.com/ishan-gaur/af3-server.git"
    ```

=== "conda / pip"

    ```bash
    pip install "af3-server @ git+https://github.com/ishan-gaur/af3-server.git"
    ```

See the [af3-server README](https://github.com/ishan-gaur/af3-server) for server setup instructions.

**ProteinMPNN** — via the Foundry package. Required for ProteinMPNN and StabilityPMPNN models:

=== "uv"

    ```bash
    uv pip install "rc-foundry[all]"
    uv run foundry install proteinmpnn
    ```

=== "conda / pip"

    ```bash
    pip install "rc-foundry[all]"
    foundry install proteinmpnn
    ```

### Verify installation

=== "uv"

    ```bash
    uv run python -c "import proteingen; print('ProteinGen installed successfully')"
    ```

=== "conda / pip"

    ```bash
    python -c "import proteingen; print('ProteinGen installed successfully')"
    ```

## 4. Configure Your Agent to Use ProteinGen

Copy this into your **project's** `AGENTS.md` or `CLAUDE.md` (whatever your agent reads). It instructs your agent to find ProteinGen and use its documentation.

```bash
## ProteinGen

Upon your first use of proteingen, complete the following tasks. First, find the path of the installed dependency and add it below. Second, find the .agents/skills/ directory in the proteingen folder and copy that into your main skills folder. The correct location will depend on your agent harness, so look it up or check with your user if unsure.

- Library location: {FILL THIS IN}
- When using ProteinGen, read `../proteingen/AGENTS.md` for working knowledge about the library's internals, gotchas, and design decisions. Make sure to recursively follow the AGENTS.md to the appropriate markdown file discussing the feature of the library you need to use for your task
- Documentation: `../proteingen/docs/` or https://ishan-gaur.github.io/proteingen/
```

The above text will instruct your model to look at markdown files we've included throughout the codebase that accumulate fixes to errors and gotchas we've found when using agents with ProteinGen over time. It also installs the agent skills ProteinGen ships with. Skills are step-by-step workflows your coding agent can follow for common tasks (e.g. adding a new generative model). See the [available skills](workflows/index.md#skills) for a full listing.


## 5. Next Steps

You're all set! Here's where to go from here:

- **[Design Philosophy](reference/design-philosophy.md)** — understand the three base classes (`ProbabilityModel`, `GenerativeModel`, `PredictiveModel`) and how they compose
- **[Examples](examples/index.md)** — working end-to-end code for sampling, fine-tuning, and guided generation
- **[Workflows](workflows/index.md)** — step-by-step recipes for common protein design tasks
- **[Models](models/index.md)** — all supported models, their capabilities, and code examples
