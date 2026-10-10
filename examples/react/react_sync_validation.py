from typing import Any

from dotenv import load_dotenv

from tenxgraph.core import Agent, StateGraph, ToolNode
from tenxgraph.core.state import AgentState, Message
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils.callbacks import BaseValidator, CallbackManager
from tenxgraph.utils.constants import END
from tenxgraph.utils.validators import (
    MessageContentValidator,
    PromptInjectionValidator,
    ValidationError,
)


load_dotenv()

checkpointer = InMemoryCheckpointer()


class BusinessPolicyValidator(BaseValidator):
    """
    Custom validator to enforce business-specific policies.

    This demonstrates how to create your own validator by extending BaseValidator.
    """

    def __init__(self, strict_mode: bool = True, max_message_length: int = 10000):
        self.strict_mode = strict_mode
        self.max_message_length = max_message_length
        self.forbidden_topics = [
            "financial advice",
            "medical diagnosis",
            "legal counsel",
        ]

    def _handle_violation(self, message: str, violation_type: str, details: dict[str, Any]) -> None:
        """Handle a validation violation."""
        print(f"[WARNING] Validation violation: {violation_type} - {message}")
        if self.strict_mode:
            raise ValidationError(message, violation_type, details)

    async def validate(self, messages: list[Message]) -> bool:
        """Validate messages according to business policies."""
        for msg in messages:
            # Use .text() method to extract text from message content
            content_str = msg.text()
            content_lower = content_str.lower()

            # Check message length
            if len(content_str) > self.max_message_length:
                self._handle_violation(
                    f"Message exceeds maximum length of {self.max_message_length} characters",
                    "message_too_long",
                    {"message_length": len(content_str), "max_length": self.max_message_length},
                )

            # Check for forbidden topics
            for topic in self.forbidden_topics:
                if topic in content_lower:
                    self._handle_violation(
                        f"Message contains forbidden topic: {topic}",
                        "forbidden_topic",
                        {"topic": topic, "content_snippet": content_lower[:100]},
                    )

            # Check for all-caps (shouting) - use original string
            MIN_CAPS_LENGTH = 10
            if content_str.isupper() and len(content_str) > MIN_CAPS_LENGTH:
                self._handle_violation(
                    "Message contains excessive capitalization",
                    "excessive_caps",
                    {"content_length": len(content_str)},
                )

        return True


# Set up callback manager with validators
callback_manager = CallbackManager()
callback_manager.register_input_validator(PromptInjectionValidator(strict_mode=True))
callback_manager.register_input_validator(MessageContentValidator())
callback_manager.register_input_validator(
    BusinessPolicyValidator(strict_mode=True, max_message_length=5000)
)


class CustomAgentState(AgentState):
    jd_name: str = "CustomAgentState"


def get_weather(
    location: str,
    tool_call_id: str | None = None,
    state: CustomAgentState | None = None,
) -> str:
    """
    Get the current weather for a specific location.
    This demo shows injectable parameters: tool_call_id and state are automatically injected.
    """
    # You can access injected parameters here
    if tool_call_id:
        print(f"Tool call ID: {tool_call_id}")
    if state and hasattr(state, "context"):
        print(f"Number of messages in context: {len(state.context)}")  # type: ignore

    # return f"The weather in {location} is sunny"
    raise Exception("Simulated tool failure for testing error handling.")


# def update_context(
#     state: CustomAgentState,
#     jd_name: str,
# ) -> ToolResult:
#     """Update the current jd name in the state and report back to the AI."""
#     return ToolResult(
#         message=f"JD name has been updated to '{jd_name}'.",
#         state={"jd_name": jd_name},
#     )


tool_node = ToolNode(
    [
        get_weather,
        # update_context,
    ]
)

# Create agent with tools
agent = Agent(
    model="gemini-3-flash-preview",
    provider="google",
    system_prompt=[
        {
            "role": "system",
            "content": """
                You are a helpful assistant.
                Your task is to assist the user in finding information and answering questions.
            """,
        },
        {"role": "user", "content": "Today Date is 2024-06-15"},
    ],
    tool_node="TOOL",
    trim_context=True,
    reasoning_config=True,
)


def should_use_tools(state: AgentState) -> str:
    """Determine if we should use tools or end the conversation."""
    if not state.context or len(state.context) == 0:
        return "TOOL"  # No context, might need tools

    last_message = state.context[-1]

    # If the last message is from assistant and has tool calls, go to TOOL
    if (
        hasattr(last_message, "tools_calls")
        and last_message.tools_calls
        and len(last_message.tools_calls) > 0
        and last_message.role == "assistant"
    ):
        return "TOOL"

    # If last message is a tool result, we should be done (AI will make final response)
    if last_message.role == "tool":
        return "MAIN"

    # Default to END for other cases
    return END


graph = StateGraph()
graph.add_node("MAIN", agent)
graph.add_node("TOOL", tool_node)

# Add conditional edges from MAIN
graph.add_conditional_edges(
    "MAIN",
    should_use_tools,
    {"TOOL": "TOOL", END: END},
)

# Always go back to MAIN after TOOL execution
graph.add_edge("TOOL", "MAIN")
graph.set_entry_point("MAIN")


app = graph.compile(
    checkpointer=checkpointer,
    callback_manager=callback_manager,
)

# now run it

inp = {"messages": [Message.text_message("Please call the get_weather function for New York City")]}
config = {"thread_id": "12345", "recursion_limit": 10}


res = app.invoke(inp, config=config)

for i in res["messages"]:
    print("**********************")
    print("Message Type: ", i.role)
    print(i)
    print("**********************")
    print("\n\n")


# grp = app.generate_graph()

# print(grp)
# res = app.stream(inp, config=config)

# for i in res:
#     print("**********************")
#     print("Message Type: ", i)
#     print(i)
#     print("**********************")
#     print("\n\n")
