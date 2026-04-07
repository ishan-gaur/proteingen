# Setup

If you find yourself uncomfortable with any of the terminal use, package management, etc. in the guide below, we recommend checking out MIT's ["Missing Semester of Your CS Education"](https://missing.csail.mit.edu/). The first three videos + the lecture on git and version control should get you largely up to speed.

## 1. (Recommended) Install a Coding Agent

<!-- TODO[pi] add links to these other agents' installation instructions -->
ProteinGen was designed to facilitate the effective use of AI coding agents with our code. We recommend [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview) (instructions at the bottom of this section). A few other options are Codex, Amp, Pi, and Gemini CLI. These tools typically require a paid plan with a model provider, but we've gotten plenty of mileage for our tasks out of the basic tiers. 

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


<!-- TODO [pi] collapse the folling section by default; turn it into an admonition https://squidfunk.github.io/mkdocs-material/reference/admonitions/ -->
### Installing Claude 

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

<!-- make this a regular collapsed section by default -->
## 2. (Optional) Install a Package Manager

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. We use it for all dependency management and running scripts. Some other options include Conda, Poetry, and good ol' venv.
<!-- TODO[pi] add links to the instructions for conda, poetry, and venv -->

!!! tip
    `pip` can often be used from within `uv`. If you see something like `uv pip install ...` you can remove uv and use the command just fine. This can be useful, for example, if you choose not to use `uv` but use `conda` instead. You can just run the pip commmand from within the conda environment. That being said, *uv is super cool, and we highly recommend you giving it a try*. If you want to know how *package management* can be an interesting problem, checkout this [talk](https://www.youtube.com/watch?v=gSKTfG1GXYQ) from the `uv` founder.

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

You can create a new project by running:

```bash
# Create your project
mkdir my-protein-project
cd my-protein-project
git init
uv init
cd ..
```

## 3. Install ProteinGen

ProteinGen is designed to be a library you own: that you and your agents can adapt as you find what works for you. Accordingly, we recommend doing an editable local install, however you can also install it as a one-off dependency if you'd like.

### Editable Local Install (Recommended)

```bash
# Clone ProteinGen
cd path/to/your/projects/folder && git clone https://github.com/ishan-gaur/proteingen.git
```
<!-- TODO[pi] add installation instructions for conda as well -->
```bash
# Install ProteinGen and (optional components)
cd path/to/your
uv pip install -e .

# Optional components
uv run foundry install proteinmpnn
cd ..

# Add ProteinGen as an editable dependency of your project
uv add --editable ./proteingen
```

## Point your agent at ProteinGen

In your **project's** `AGENTS.md` (or `CLAUDE.md`, or whatever your agent reads) tell your agent where to find ProteinGen and that it should use its documentation:



```bash
cat >> AGENTS.md << 'EOF'

## ProteinGen

- Library location: `./proteingen/` (installed as editable package)
- When using ProteinGen, read `./proteingen/AGENTS.md` for working knowledge about the library's internals, gotchas, and design decisions.
- Documentation: `./proteingen/docs/` or https://ishan-gaur.github.io/proteingen/
EOF
```

### Add agent skills

ProteinGen ships with agent skills in `proteingen/.agents/skills/` — step-by-step workflows your coding agent can follow for common tasks (e.g. adding a new generative model). See the [available skills](workflows/index.md#skills) for a full listing. Copy or symlink them to wherever your agent harness expects skills. Since you have an agent, we recommend just asking it to move the skills itself:

> Can you locate where the proteingen package is installed in my {uv, conda, etc.} environment? Then copy the skills inside of proteingen/.agents/skills/ to wherever you store your skills normally?

If you want to do it manually, run:
```bash
cp -r /path/to/proteingen/.agents/skills/ /path/to/agent/skills/
```

Where `path/to/proteingen` might depend on how you installed ProteinGen:

=== "Editable Install (Git Cloned)"

    ```bash
    /path/to/where/you/cloned/proteingen
        
    ```

=== "uv"

    ```bash
    # from within your uv project directory (ie /path/to/my-protein-project)
    # first * is a wildcard for the python type
    # second * is a wildcard for the package version you're installing
    .venv/lib/*/site-packages/proteingen*/.agents/skills/
    ```

<!-- TODO[pi] add the instructions for conda -->

and `/path/to/agent/skills/` depends on the coding agent you're using:

=== "Claude Code"
    
    ```bash
    # in your my-protein-project directory
    ~/.pi/skills/
    ```

=== "Pi"

    ```bash
    ~/.pi/skills/
    ```

=== "Other agents"

    Check your agent's docs for where it discovers skills, then copy the `proteingen/.agents/skills/` directory there.

But seriously, your agent can probably just do this for you if you do feel familiar with the terminal and all this installation stuff.

<!-- TODO[pi]: finalize the recommended project setup flow — should we provide a `proteingen init` CLI command? -->

## 4. Learn the design

Before writing code, read the [Design Philosophy](reference/design-philosophy.md) section to understand the three base classes:

- **`ProbabilityModel`** — shared base for all models (temperature, conditioning, log-probs)
- **`GenerativeModel`** — wraps generative models (masked LMs, flow matching)
- **`PredictiveModel`** — wraps predictive models ("does this sequence have property X?")

These classes share a common interface for temperature and conditioning:

```python
from proteingen.models import ESMC

model = ESMC()

# Temperature control
model.set_temp_(2.0)               # in-place
model = model.set_temp(2.0)        # returns self
with model.with_temp(2.0):         # context manager (reverts after)
    probs = model.get_log_probs(x)

# Conditioning
model.set_condition_({"coords_RAX": coords})   # in-place
with model.conditioned_on({"coords_RAX": coords}):  # context manager
    probs = model.get_log_probs(x)
```

Then check out the [Examples](examples/index.md) to see working end-to-end code.
