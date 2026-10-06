"""
Personalized AI Agent using TAF + Mem0 + Cloud Qdrant

This example demonstrates a sophisticated personalized AI agent that:
1. Uses TAF framework for agent orchestration
2. Integrates Mem0 for advanced memory management
3. Uses Cloud Qdrant for vector storage (768 dimensions for Gemini embeddings)
4. Maintains conversation context and user preferences

Setup Required:
- GOOGLE_API_KEY: For Gemini models (LLM + embeddings)
- MEM0_API_KEY: For Mem0 cloud service
- QDRANT_URL: Your Qdrant cloud URL
- QDRANT_API_KEY: Your Qdrant cloud API key

Example .env file:
GOOGLE_API_KEY=your_google_api_key
MEM0_API_KEY=your_mem0_api_key
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
"""

import asyncio
import os
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from mem0 import Memory

from tenxgraph.core import Agent, StateGraph
from tenxgraph.core.state import AgentState, Message
from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.utils.constants import END


# Load environment variables
load_dotenv()

# Configure environment
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
os.environ["MEM0_API_KEY"] = os.getenv("MEM0_API_KEY", "")


class PersonalizedAgentState(AgentState):
    """Extended state for personalized agent with user context."""

    user_id: str = ""
    user_profile: dict[str, Any] = {}
    session_id: str = ""
    conversation_summary: str = ""
    interaction_count: int = 0
    memory_context: str = ""
    preferences_context: str = ""


class PersonalizedAgent:
    """
    A sophisticated personalized AI agent using TAF + Mem0 + Cloud Qdrant.

    Features:
    - Maintains long-term memory across conversations
    - Adapts responses based on user preferences and history
    - Uses semantic memory search for relevant context retrieval
    - Supports multiple conversation threads per user
    """

    def __init__(self):
        """Initialize the personalized agent with Mem0 and cloud Qdrant."""

        # Mem0 configuration for cloud Qdrant
        self.mem0_config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "personalized_agent_memory",
                    "url": os.getenv("QDRANT_URL"),
                    "api_key": os.getenv("QDRANT_API_KEY"),
                    "embedding_model_dims": 768,  # Gemini embedding dimension (768 dimensions)
                },
            },
            "llm": {
                "provider": "gemini",
                "config": {
                    "model": "gemini-2.0-flash-exp",
                    "temperature": 0.1,
                    "max_tokens": 1000,
                    "top_p": 0.9,
                },
            },
            "embedder": {"provider": "gemini", "config": {"model": "models/text-embedding-004"}},
        }

        # Initialize Mem0 with cloud Qdrant
        self.memory = Memory.from_config(self.mem0_config)
        self.app_id = "personalized-agent-v1"

        # Initialize TAF graph
        self.checkpointer = InMemoryCheckpointer()
        self._build_agent_graph()

    def _build_agent_graph(self):
        """Build the TAF agent graph with memory integration."""

        # Create the response agent with state-interpolated system prompt
        self.response_agent = Agent(
            model="gemini-2.0-flash",
            provider="google",
            system_prompt=[
                {
                    "role": "system",
                    "content": """You are a highly personalized AI assistant. You maintain context across conversations and adapt to user preferences.

User ID: {user_id}
Interaction Count: {interaction_count}

{memory_context}
{preferences_context}

Instructions:
1. Reference relevant past interactions when appropriate
2. Adapt your communication style based on user preferences
3. Be helpful, engaging, and maintain conversation continuity
4. If this is a new user, introduce yourself warmly
5. Always provide thoughtful, context-aware responses

Current conversation:""",
                },
            ],
            trim_context=True,
            temperature=0.7,
            max_tokens=1500,
        )

        # Create state graph
        self.graph = StateGraph[PersonalizedAgentState](PersonalizedAgentState())

        # Add nodes
        self.graph.add_node("memory_retrieval", self._memory_retrieval_node)
        self.graph.add_node("personalized_response", self.response_agent)
        self.graph.add_node("memory_storage", self._memory_storage_node)

        # Define flow
        self.graph.set_entry_point("memory_retrieval")
        self.graph.add_edge("memory_retrieval", "personalized_response")
        self.graph.add_edge("personalized_response", "memory_storage")
        self.graph.add_edge("memory_storage", END)

        # Compile the graph
        self.app = self.graph.compile(checkpointer=self.checkpointer)

    async def _memory_retrieval_node(self, state: PersonalizedAgentState) -> PersonalizedAgentState:
        """Retrieve relevant memories and user context."""

        if not state.context:
            return state

        user_message = state.context[-1]
        user_id = state.user_id

        try:
            # Search for relevant memories
            memories = self.memory.search(
                query=user_message.text(),
                user_id=user_id,
                limit=5,
            )

            # Extract user profile and preferences from memories
            memory_context = []
            preferences = {}

            if "results" in memories:
                for memory_item in memories["results"]:
                    memory_text = memory_item.get("memory", "")
                    memory_context.append(memory_text)

                    # Extract preferences (simple pattern matching)
                    if any(
                        word in memory_text.lower()
                        for word in ["prefer", "like", "favorite", "enjoy"]
                    ):
                        preferences[f"preference_{len(preferences)}"] = memory_text

            # Build context strings for state interpolation
            if memory_context:
                state.memory_context = (
                    "Based on our previous conversations, I remember:\n"
                    + "\n".join([f"• {memory}" for memory in memory_context[:3]])
                )
                state.conversation_summary = " | ".join(memory_context[:2])
            else:
                state.memory_context = ""
                state.conversation_summary = ""

            if preferences:
                state.preferences_context = "\nYour preferences: " + "; ".join(preferences.values())
            else:
                state.preferences_context = ""

            # Update state with retrieved context
            state.user_profile = {
                "recent_memories": memory_context[:3],  # Top 3 most relevant
                "preferences": preferences,
                "memory_count": len(memory_context),
            }

            print(f"🧠 Retrieved {len(memory_context)} relevant memories for user {user_id}")

        except Exception as e:
            print(f"❌ Error retrieving memories: {e}")
            # Continue without memory context
            state.user_profile = {"error": str(e)}
            state.memory_context = ""
            state.preferences_context = ""

        return state

    async def _memory_storage_node(self, state: PersonalizedAgentState) -> PersonalizedAgentState:
        """Store the interaction in long-term memory."""

        # Increment interaction count
        state.interaction_count += 1

        if len(state.context) < 2:
            return state

        user_message = state.context[-2]  # User's message
        ai_message = state.context[-1]  # AI's response

        try:
            # Prepare interaction for memory storage
            interaction = [
                {
                    "role": "user",
                    "content": user_message.content,
                    "timestamp": datetime.now().isoformat(),
                },
                {
                    "role": "assistant",
                    "content": ai_message.content,
                    "timestamp": datetime.now().isoformat(),
                },
            ]

            # Store in Mem0 with metadata
            metadata = {
                "app_id": self.app_id,
                "session_id": state.session_id,
                "interaction_count": state.interaction_count,
                "user_id": state.user_id,
            }

            result = self.memory.add(
                messages=interaction,
                user_id=state.user_id,
                metadata=metadata,
            )

            stored_count = len(result.get("results", [])) if isinstance(result, dict) else 0
            print(f"💾 Stored interaction: {stored_count} memories added")

        except Exception as e:
            print(f"❌ Error storing memories: {e}")

        return state

    async def chat(self, user_input: str, user_id: str, session_id: str) -> str:
        """
        Main chat interface for the personalized agent.

        Args:
            user_input: The user's message
            user_id: Unique identifier for the user
            session_id: Session identifier

        Returns:
            The agent's response
        """

        if not session_id:
            session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Configure conversation thread
        config = {
            "thread_id": session_id,
            "configurable": {"user_id": user_id, "session_id": session_id},
        }

        inp = {"messages": [Message.text_message(user_input, role="user")]}

        # Run the agent
        result = await self.app.ainvoke(inp, config=config)

        # Return the AI's response
        return (
            result["messages"][-1].text()
            if result["messages"]
            else "I couldn't process your request."
        )

    def get_user_memories(self, user_id: str, limit: int = 10) -> list[dict]:
        """Get all memories for a specific user."""
        try:
            memories = self.memory.get_all(
                user_id=user_id,
            )
            return memories.get("results", [])[:limit] if isinstance(memories, dict) else []
        except Exception as e:
            print(f"❌ Error retrieving user memories: {e}")
            return []

    def delete_user_memories(self, user_id: str) -> bool:
        """Delete all memories for a specific user."""
        try:
            self.memory.delete_all(user_id=user_id)
            print(f"🗑️ Deleted all memories for user {user_id}")
            return True
        except Exception as e:
            print(f"❌ Error deleting memories: {e}")
            return False


async def demo_conversation():
    """Demo conversation showing personalized responses."""

    print("🚀 Initializing Personalized AI Agent with Mem0 + Qdrant...")
    agent = PersonalizedAgent()

    # Demo user
    user_id = "demo_user_123"
    session_id = f"demo_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n👤 Demo User: {user_id}")
    print(f"📝 Session: {session_id}")
    print("\n" + "=" * 60)

    # Conversation flow
    conversations = [
        "Hi! I'm interested in learning about machine learning. Can you help me?",
        "I prefer hands-on learning rather than just theory. What do you suggest?",
        "I work as a software engineer, so I have programming experience in Python.",
        "What are some good beginner ML projects I could start with?",
        "Thanks! Can you remind me what we discussed about my learning preferences?",
    ]

    for i, user_input in enumerate(conversations, 1):
        print(f"\n💬 User: {user_input}")

        # Get agent response
        response = await agent.chat(user_input, user_id, session_id)
        print(f"🤖 Agent: {response}")

        # Add delay between messages
        await asyncio.sleep(1)

        if i < len(conversations):
            print("\n" + "-" * 40)

    print("\n" + "=" * 60)

    # Show stored memories
    print("\n🧠 Stored Memories:")
    memories = agent.get_user_memories(user_id)
    for i, memory in enumerate(memories[:5], 1):
        print(f"{i}. {memory.get('memory', 'N/A')}")

    print(f"\nTotal memories: {len(memories)}")


def interactive_mode():
    """Interactive chat mode."""

    print("🚀 Starting Personalized AI Agent...")
    print("💡 Tip: Your conversations are remembered across sessions!")
    print("Type 'quit' to exit, 'memories' to see your stored memories\n")

    agent = PersonalizedAgent()

    # Get user ID
    user_id = input("👤 Enter your user ID (or press Enter for 'user_001'): ").strip()
    if not user_id:
        user_id = "user_001"

    session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n✅ Logged in as: {user_id}")
    print(f"📝 Session: {session_id}")
    print("\n" + "=" * 50)

    while True:
        try:
            user_input = input("\n💬 You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "bye"]:
                print("\n👋 Agent: Goodbye! I'll remember our conversation for next time.")
                break

            if user_input.lower() == "memories":
                print("\n🧠 Your stored memories:")
                memories = agent.get_user_memories(user_id)
                if memories:
                    for i, memory in enumerate(memories[:10], 1):
                        print(f"{i}. {memory.get('memory', 'N/A')}")
                    print(f"\nShowing {min(10, len(memories))} of {len(memories)} memories")
                else:
                    print("No memories found.")
                continue

            # Get agent response
            print("🤖 Agent: ", end="", flush=True)
            response = asyncio.run(agent.chat(user_input, user_id, session_id))
            print(response)

        except KeyboardInterrupt:
            print("\n\n👋 Agent: Goodbye! I'll remember our conversation for next time.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.")


if __name__ == "__main__":
    import sys

    # Check for required environment variables
    required_vars = ["GOOGLE_API_KEY", "MEM0_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these in your .env file and try again.")
        sys.exit(1)

    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        # Run demo conversation
        asyncio.run(demo_conversation())
    else:
        # Run interactive mode
        interactive_mode()
