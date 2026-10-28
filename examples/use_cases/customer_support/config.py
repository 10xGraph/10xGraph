"""Settings for the support demo. Only API keys come from the environment."""

from __future__ import annotations

from pathlib import Path


HERE = Path(__file__).parent

# Refunds above this amount (USD) pause the run for human approval.
REFUND_APPROVAL_THRESHOLD = 50.0

# Conversation state is saved here, so closing the CLI and starting it again
# with the same customer id resumes the same thread.
CHECKPOINT_DB = HERE / "data" / "support_checkpoints.db"

# Local, file-based Qdrant: no Docker or server needed.
QDRANT_PATH = HERE / "data" / "qdrant"
MEMORY_COLLECTION = "support_customer_memory"
