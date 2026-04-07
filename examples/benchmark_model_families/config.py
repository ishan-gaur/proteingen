"""Shared configuration for the model families benchmark."""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

# ── Experiment parameters ────────────────────────────────────────────────
N_SWISSPROT_SEQS = 10
N_ORDERS = 5  # random decoding orders per protein
MASK_FRACTIONS = [0.10, 0.25, 0.50, 1.00]
RNG_SEED = 42

# Length range for SwissProt sequences (tokenized length incl. BOS/EOS)
MIN_SEQ_LEN = 80
MAX_SEQ_LEN = 300

# ── Model definitions ───────────────────────────────────────────────────
# Each entry: (family, display_name, constructor, approx_param_count)
# Constructor is a string so we can selectively import in generate.py.
MODEL_CONFIGS = [
    # ESM-C family
    ("esmc", "ESMC-300M", "esmc_300m", 300),
    ("esmc", "ESMC-600M", "esmc_600m", 600),
    # ESM3
    ("esm3", "ESM3-Open", "esm3-open", 1400),
    # DPLM2 family
    ("dplm2", "DPLM2-150M", "airkingbd/dplm2_150m", 150),
    ("dplm2", "DPLM2-650M", "airkingbd/dplm2_650m", 650),
    ("dplm2", "DPLM2-3B", "airkingbd/dplm2_3b", 3000),
]

# ── AF3 server ──────────────────────────────────────────────────────────
AF3_SERVER_URL = "http://localhost:8080"
AF3_POLL_INTERVAL = 10.0
AF3_TIMEOUT = 7200.0  # 2 hours per job

# ── Fold class definitions ──────────────────────────────────────────────
# Broad SCOP-like fold classes based on secondary structure content
FOLD_CLASSES = {
    "all-alpha": lambda h, s: h >= 0.40 and s < 0.10,
    "all-beta": lambda h, s: h < 0.10 and s >= 0.30,
    "alpha+beta": lambda h, s: h >= 0.15 and s >= 0.15,
    "small/other": lambda h, s: True,  # fallback
}
