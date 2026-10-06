"""
Simple Personalized Agent Example using TAF + Mem0 + Cloud Qdrant

A streamlined example showing basic integration between:
- TAF for agent framework
- Mem0 for memory management
- Cloud Qdrant for vector storage

This example demonstrates a chatbot that remembers user preferences and conversation history.
"""

import asyncio
import os

from dotenv import load_dotenv
from mem0 import Memory

from tenxgraph.core import Agent, StateGraph
from tenxgraph.core.state import AgentState, Message
from tenxgraph.utils.constants import END


# Load environment variables
load_dotenv()

# Set environment variables
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
os.environ["MEM0_API_KEY"] = os.getenv("MEM0_API_KEY", "")


class MemoryAgentState(AgentState):
    """State with user ID and memory context for interpolation."""

    user_id: str = ""
    memory_context: str = ""


class SimplePersonalizedAgent:
    """Simple personalized agent using Mem0 for memory."""

    def __init__(self):
        # Mem0 configuration for cloud Qdrant (768 dimensions for Gemini)
        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "simple_agent_memory",
                    "url": os.getenv("QDRANT_URL"),
                    "api_key": os.getenv("QDRANT_API_KEY"),
                    "embedding_model_dims": 768,  # For Gemini embeddings (768 dimensions)
                },
            },
            "llm": {
                "provider": "gemini",
                "config": {"model": "gemini-2.0-flash-exp", "temperature": 0.1},
            },
            "embedder": {"provider": "gemini", "config": {"model": "models/text-embedding-004"}},
        }

        self.memory = Memory.from_config(config)
        self.app_id = "simple-agent"
        self._build_graph()

    def _build_graph(self):
        """Build TAF graph with memory integration."""
        # Create the response agent with state-interpolated system prompt
        self.response_agent = Agent(
            model="gemini-2.0-flash",
            provider="google",
            system_prompt=[
                {
                    "role": "system",
                    "content": """You are a helpful AI assistant with memory of past conversations.

{memory_context}

Be conversational, helpful, and reference past interactions when relevant.""",
                },
            ],
            trim_context=True,
        )

        graph = StateGraph[MemoryAgentState](MemoryAgentState())

        graph.add_node("memory_retrieval", self._memory_retrieval_node)
        graph.add_node("chat", self.response_agent)
        graph.add_node("memory_storage", self._memory_storage_node)

        graph.set_entry_point("memory_retrieval")
        graph.add_edge("memory_retrieval", "chat")
        graph.add_edge("chat", "memory_storage")
        graph.add_edge("memory_storage", END)

        self.app = graph.compile()

    async def _memory_retrieval_node(self, state: MemoryAgentState) -> MemoryAgentState:
        """Retrieve relevant memories and build context for interpolation."""
        if not state.context:
            return state

        user_message = state.context[-1].text()
        user_id = state.user_id

        memories = []
        try:
            memory_results = self.memory.search(
                query=user_message,
                user_id=user_id,
                limit=3,
            )

            if "results" in memory_results:
                memories = [m["memory"] for m in memory_results["results"]]
            print(f"Retrieved {len(memories)} memories")
        except Exception as e:
            print(f"Memory retrieval error: {e}")

        # Build context string for state interpolation
        if memories:
            state.memory_context = "Relevant memories:\n" + "\n".join([f"- {m}" for m in memories])
        else:
            state.memory_context = ""

        return state

    async def _memory_storage_node(self, state: MemoryAgentState) -> MemoryAgentState:
        """Store the interaction in long-term memory."""
        if len(state.context) < 2:
            return state

        user_message = state.context[-2]
        ai_message = state.context[-1]

        try:
            interaction = [
                {"role": "user", "content": user_message.content},
                {"role": "assistant", "content": ai_message.content},
            ]

            self.memory.add(
                messages=interaction, user_id=state.user_id, metadata={"app_id": self.app_id}
            )
            print(f"✅ Memory stored for user {state.user_id}")
        except Exception as e:
            print(f"Memory storage error: {e}")

        return state

    async def chat(self, message: str, user_id: str) -> str:
        """Simple chat interface."""
        inp = {
            "messages": [Message.text_message(message, role="user")],
        }
        config = {
            "thread_id": user_id,
            "recursion_limit": 10,
            "user_id": user_id,
        }
        result = await self.app.ainvoke(inp, config=config)
        return result["messages"][-1].text()


# Example usage
async def main():
    """Example conversation."""
    agent = SimplePersonalizedAgent()
    user_id = "test_user"

    # Conversation examples
    conversations = [
        "Hi, I'm John and I love pizza!",
        "What are some good pizza toppings?",
        "What do you remember about my food preferences?",
        "I also enjoy hiking on weekends",
        "What activities do I enjoy based on our conversation?",
    ]

    print("🤖 Simple Personalized Agent Demo\n")

    for i, message in enumerate(conversations, 1):
        print(f"👤 User: {message}")
        response = await agent.chat(message, user_id)
        print(f"🤖 Agent: {response}\n")

        if i < len(conversations):
            await asyncio.sleep(1)  # Brief pause


if __name__ == "__main__":
    asyncio.run(main())
