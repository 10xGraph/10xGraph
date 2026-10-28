"""Ready-made tools: safe_calculator, fetch_url and sandboxed file tools.

Pass them to an agent like your own functions. create_file_tools() fixes the
directory the file tools can touch, and hides secret files such as .env.

Run: python examples/tools/prebuilt_tools.py   (needs DEEPSEEK_API_KEY and network)
"""

import asyncio
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from tenxgraph.core.state import Message
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.prebuilt.tools import create_file_tools, fetch_url, safe_calculator


load_dotenv()


async def main() -> None:
    workspace = Path(tempfile.mkdtemp(prefix="tools-demo-"))
    (workspace / "notes.txt").write_text("Budget: 1200 EUR\nTravel: 340 EUR\n")

    agent = ReactAgent(
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        system_prompt=[{"role": "system", "content": "Use tools. Be brief."}],
        tools=[
            safe_calculator,  # AST-based arithmetic, no eval()
            fetch_url,  # public http(s) only; private hosts are blocked, output truncated
            # read/write/search limited to `workspace`; write=False makes it read-only.
            *create_file_tools(workspace, write=False),
        ],
    )
    async with agent.compile() as app:
        prompt = "Read notes.txt, subtract the travel cost from the budget, and give the result."
        result = await app.ainvoke(
            {"messages": [Message.text_message(prompt)]},
            config={"thread_id": "prebuilt-1"},
        )
        print("assistant:", result["messages"][-1].text())


if __name__ == "__main__":
    asyncio.run(main())
