# Setup

## 1. Install Claude Code

ProteinGen is designed to be used with an AI coding agent. We recommend [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview):

```bash
npm install -g @anthropic-ai/claude-code
```

<!-- TODO[pi]: add brief note on alternative agents once we test compatibility -->

## 2. Install uv

We use [uv](https://docs.astral.sh/uv/) for Python package management:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 3. Set up your project

Create a repo for your protein design project, then have Claude clone ProteinGen and wire it up:

```bash
# Create your project
mkdir my-protein-project && cd my-protein-project
git init
uv init

# Clone ProteinGen as a dependency
git clone https://github.com/ishan-gaur/proteingen.git
cd proteingen && uv sync && uv pip install -e . && uv run foundry install proteinmpnn && cd ..

# Add ProteinGen to your project's dependencies
uv add --editable ./proteingen
```

Then add ProteinGen's location to your project's `AGENTS.md` so that Claude (or any agent) knows where to find the library and its documentation:

```bash
cat >> AGENTS.md << 'EOF'

## ProteinGen

- Library location: `./proteingen/` (installed as editable package)
- When using ProteinGen, read `./proteingen/AGENTS.md` for working knowledge about the library's internals, gotchas, and design decisions.
- Documentation: `./proteingen/docs/` or https://ishan-gaur.github.io/proteingen/
EOF
```

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

Then check out the [Examples](examples.md) to see working end-to-end code.
