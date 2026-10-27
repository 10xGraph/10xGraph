"""A tool that updates a custom state field by returning ToolResult.

ToolResult(message=..., state={...}) sends `message` back to the model and writes the
listed fields into the graph state. Fields must already exist on your AgentState subclass.

Run: python examples/tools/tool_result_state.py   (needs DEEPSEEK_API_KEY)
"""

import asyncio
import os

from dotenv import load_dotenv
from pydantic import Field

from tenxgraph.core.state import AgentState, Message, ToolResult
from tenxgraph.prebuilt.agent import ReactAgent
from tenxgraph.utils.constants import ResponseGranularity


load_dotenv()


class CartState(AgentState):
    cart: list[str] = Field(default_factory=list)
    total: float = 0.0


def add_to_cart(item: str, price: float, state: CartState) -> ToolResult:
    """Add an item with a price to the shopping cart."""
    # Read the current state, then return only the fields that change.
    cart = [*state.cart, item]
    total = round(state.total + price, 2)
    return ToolResult(
        message=f"Added {item}. Cart has {len(cart)} items, total {total}.",
        state={"cart": cart, "total": total},
    )


agent = ReactAgent(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    state=CartState(),
    system_prompt=[{"role": "system", "content": "Use add_to_cart for every item the user lists."}],
    tools=[add_to_cart],
)


async def main() -> None:
    async with agent.compile() as app:
        result = await app.ainvoke(
            {"messages": [Message.text_message("Add a keyboard at 49.90 and a mouse at 19.50.")]},
            config={"thread_id": "cart-1"},
            # FULL returns the final state, where the tool's updates are visible.
            response_granularity=ResponseGranularity.FULL,
        )
        state = result["state"]
        print("cart:", state.cart, "total:", state.total)
        print("assistant:", result["messages"][-1].text())


if __name__ == "__main__":
    asyncio.run(main())
