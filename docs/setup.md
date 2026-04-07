# Setup

## 1. Install uv

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. We use it for all dependency management and running scripts.

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

## 2. (Optional) Install Claude Code


ProteinGen was designed to facilitate the effective use of AI coding agents with our code. We recommend [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview). These tools typically require a paid plan, but we've gotten plenty of mileage for our tasks out of the basic tiers.



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

## 3. Set up your project

ProteinGen is meant to be **cloned into your own project and modified**. It's a library you own — adapt models, add new ones, tweak sampling algorithms, and change whatever you need to match your use case. Your coding agent can read the codebase, understand the design, and help you make changes confidently.

### Create your project and clone ProteinGen

```bash
# Create your project
mkdir my-protein-project && cd my-protein-project
git init
uv init

# Clone ProteinGen into your project
git clone https://github.com/ishan-gaur/proteingen.git
```

### Install ProteinGen and its dependencies

```bash
cd proteingen
uv sync
uv pip install -e .
uv run foundry install proteinmpnn
cd ..

# Add ProteinGen as an editable dependency of your project
uv add --editable ./proteingen
```

### Point your agent at ProteinGen

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

ProteinGen ships with agent skills in `proteingen/.agents/skills/` — step-by-step workflows your coding agent can follow for common tasks (e.g. adding a new generative model). See the [available skills](workflows/index.md#skills) for a full listing. Copy or symlink them to wherever your agent harness expects skills:

=== "Claude Code"

    ```bash
    cp -r ./proteingen/.agents/skills/ .claude/skills/
    ```

=== "Pi"

    ```bash
    cp -r ./proteingen/.agents/skills/ .pi/skills/
    ```

=== "Other agents"

    Check your agent's docs for where it discovers skills, then copy the `proteingen/.agents/skills/` directory there.

<!-- TODO[pi]: finalize the recommended project setup flow — should we provide a `proteingen init` CLI command? -->

## 4. Learn the design

Before writing code, read the [Design Philosophy](reference/design-philosophy.md) section to understand the three base classes:

- **`ProbabilityModel`** — shared base for all models (temperature, conditioning, log-probs)
- **`TransitionModel`** — wraps generative models (masked LMs, flow matching)
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
