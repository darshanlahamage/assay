"""The Auditor: independent detection, interpretability, and the experiment ledger.

Verdict comes from deterministic anchors (hidden-test gap, file hashes, the
independent tool log). AST/CoT/logprob/probe signals explain the verdict; they
never produce it.
"""

from assay.audit.ledger import Ledger, LedgerRow
from assay.audit.monitor import Auditor

__all__ = ["Auditor", "Ledger", "LedgerRow"]
