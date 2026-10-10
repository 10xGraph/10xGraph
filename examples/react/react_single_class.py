from dotenv import load_dotenv

from tenxgraph.core.state import AgentState, Message
from tenxgraph.prebuilt.agent import ReactAgent


load_dotenv()


def get_weather(
    location: str,
    tool_call_id: str | None = None,
    state: AgentState | None = None,
) -> str:
    if tool_call_id:
        print(f"tool_call_id={tool_call_id}")
    if state is not None:
        print(f"context_messages={len(state.context)}")
    return f"The weather in {location} is sunny."


react_agent = ReactAgent(
    model="google/gemini-2.5-flash",
    provider="google",
    system_prompt=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Use tools when they help answer the user.",
        }
    ],
    tools=[get_weather],
    trim_context=True,
)


if __name__ == "__main__":
    app = react_agent.compile()

    result = app.invoke(
        {"messages": [Message.text_message("Check the weather in Tokyo and answer normally.")]},
        config={"thread_id": "react-single-class", "recursion_limit": 10},
    )

    for message in result["messages"]:
        print("**********************")
        print("Message Type:", message.role)
        print(message)
        print("**********************")
