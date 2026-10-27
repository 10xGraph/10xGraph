"""Research assistant: plan, search, reflect, then write a structured report.

Claude plans, GPT-5 mini searches and reads pages, Gemini reviews, DeepSeek writes
the JSON report. Progress streams to the terminal and the report is saved as markdown.

Run from the repo root:
    python examples/use_cases/research_assistant/main.py "How do solid-state batteries work?"
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from graph import build_graph
from report import parse_report, save_markdown, to_markdown

from tenxgraph.core.state import Message
from tenxgraph.core.state.stream_chunks import StreamEvent


load_dotenv()

OUTPUT = Path(__file__).parent / "output" / "report.md"
DEFAULT_QUESTION = "What are the main trade-offs between solid-state and lithium-ion batteries?"


async def main() -> None:
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    app = build_graph()
    node = ""
    deltas: list[str] = []  # REPORT text, collected instead of printed
    final_report = ""

    async for chunk in app.astream({"messages": [Message.text_message(question)]}):
        if chunk.event == StreamEvent.UPDATES and chunk.data.get("status") == "node_invoked":
            node = chunk.data["node"]
            print(f"\n--- {node} ---")
        elif chunk.event == StreamEvent.ERROR:
            print("error:", chunk.data)
        elif chunk.event == StreamEvent.MESSAGE and chunk.message is not None:
            msg = chunk.message
            if node == "REPORT":
                if msg.delta:
                    deltas.append(msg.text())
                elif msg.role == "assistant":
                    final_report = msg.text()
            elif msg.delta:
                print(msg.text(), end="", flush=True)
            elif msg.tools_calls:
                names = [c["function"]["name"] for c in msg.tools_calls]
                print(f"\n[tool calls] {', '.join(names)}")

    report = parse_report(final_report or "".join(deltas))
    path = save_markdown(report, OUTPUT)
    print("\n" + to_markdown(report))
    print(f"saved: {path}")
    await app.aclose()


if __name__ == "__main__":
    asyncio.run(main())
