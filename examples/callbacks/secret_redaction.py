"""Secret redaction: keep API keys and tokens out of your logs.

mask_secrets() scrubs credential patterns from a string. install_secret_redaction()
attaches the same filter to the tenxgraph logger so log records are scrubbed too.
No model call is needed, so this runs offline.

Run: python examples/callbacks/secret_redaction.py
"""

import asyncio
import logging

from tenxgraph.utils import install_secret_redaction, mask_secrets


async def main() -> None:
    samples = [
        "Calling OpenAI with key sk-proj-abcdefghijklmnopqrstuvwxyz123456",
        "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.signature",
        "retrying with api_key=super-secret-value",
        "no secrets in this line",
    ]
    print("mask_secrets:")
    for line in samples:
        print("  in :", line)
        print("  out:", mask_secrets(line))

    # Handlers added to the logger are covered too. Configure handlers first, then install.
    logger = logging.getLogger("tenxgraph")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("log> %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    install_secret_redaction("tenxgraph")

    print("\nlogger output:")
    logger.info("request headers: Authorization: Bearer abc123def456ghi789")


if __name__ == "__main__":
    asyncio.run(main())
