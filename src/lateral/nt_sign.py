"""Neurotransmitter-based sign classification for connectome edges.

Drosophila inhibitory transmitters: ``GABA`` (GABA-A, Cl-), ``HIS`` (histamine via
HisCl1; the R1-6 photoreceptors) and ``GLUT`` (glutamate, frequently inhibitory via
GluCl). ``ACH`` (acetylcholine) is excitatory. The remainder (DA / SER / OCT) are
modulatory and mapped to ``"other"``.

Operational caveat carried over from the notebooks: ``nt_type`` is mostly
ML-predicted (only R1-6 is domain-corrected to ``HIS``), and a connectome shows
wiring -- not function. The true functional sign is receptor-dependent, so trust
aggregates rather than individual edges. A probabilistic-sign robustness layer
(using ``gaba_avg / glut_avg / ach_avg`` + ``nt_type_score``) is intentionally
deferred to a follow-up issue; this module is the deterministic baseline only.
"""

from __future__ import annotations

INHIBITORY_NT = frozenset({"GABA", "GLUT", "HIS"})
EXCITATORY_NT = frozenset({"ACH"})

# Numeric sign used for multiplicative path arithmetic (see pathtrace.py).
# ``other`` is 0 so any path through a modulatory edge drops out of net-sign metrics.
SIGN_VALUE = {"exc": 1, "inh": -1, "other": 0}


def classify_nt(value) -> str:
    """Map an ``nt_type`` label to one of ``{"inh", "exc", "other"}``."""
    if value in INHIBITORY_NT:
        return "inh"
    if value in EXCITATORY_NT:
        return "exc"
    return "other"


def add_sign(conn_df, *, nt_col="nt_type", out_col="sign"):
    """Return a copy of ``conn_df`` with ``out_col`` = ``classify_nt(nt_col)``."""
    out = conn_df.copy()
    out[out_col] = out[nt_col].map(classify_nt)
    return out
