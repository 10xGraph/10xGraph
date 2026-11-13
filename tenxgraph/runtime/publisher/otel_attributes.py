"""OTEL attribute name constants aligned with GenAI semantic conventions.

https://opentelemetry.io/docs/specs/semconv/gen-ai/
"""

# ── gen_ai core ──────────────────────────────────────────────────────────────
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_OPERATION = "gen_ai.operation.name"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"

# ── gen_ai tool attributes (execute_tool spans) ──────────────────────────────
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
GEN_AI_TOOL_TYPE = "gen_ai.tool.type"

# ── gen_ai request attributes (standard semconv) ─────────────────────────────
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
GEN_AI_REQUEST_TOP_K = "gen_ai.request.top_k"
GEN_AI_REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
GEN_AI_REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
GEN_AI_REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"
GEN_AI_REQUEST_SEED = "gen_ai.request.seed"

# ── gen_ai response attributes (standard semconv) ────────────────────────────
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_RESPONSE_ID = "gen_ai.response.id"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"

# ── gen_ai usage — base (STANDARD level) ────────────────────────────────────
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

# Token counts on graph and node spans. Kept off gen_ai.usage.* so backends that sum
# descendant spans count only the LLM spans.
USAGE_INPUT_TOKENS = "tenxgraph.usage.input_tokens"
USAGE_OUTPUT_TOKENS = "tenxgraph.usage.output_tokens"

# ── gen_ai usage — extended token types ──────────────────────────────────────
# Prompt caching (Anthropic + OpenAI)
GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens"
GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS = "gen_ai.usage.cache_creation.input_tokens"
# Extended thinking / reasoning (OpenAI o-series, Claude thinking)
GEN_AI_USAGE_REASONING_OUTPUT_TOKENS = "gen_ai.usage.reasoning.output_tokens"

# ── gen_ai I/O content (FULL level, as span event attributes) ────────────────
# Langfuse and most OTEL backends read these from span events.
GEN_AI_PROMPT = "gen_ai.prompt"
GEN_AI_COMPLETION = "gen_ai.completion"
# Modern semconv attribute names (preferred in newer Langfuse versions)
GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"

# ── session / trace context ───────────────────────────────────────────────────
SESSION_ID = "session.id"

# ── graph (invoke_agent) span attributes ────────────────────────────────────
GRAPH_THREAD_ID = "tenxgraph.graph.thread_id"
GRAPH_RUN_ID = "tenxgraph.graph.run_id"
GRAPH_USER_ID = "tenxgraph.graph.user_id"
GRAPH_TOTAL_STEPS = "tenxgraph.graph.total_steps"
GRAPH_MODEL = "tenxgraph.graph.model"

# ── node span attributes ─────────────────────────────────────────────────────
NODE_NAME = "tenxgraph.node.name"
NODE_STEP = "tenxgraph.node.step"

# ── tool (execute_tool) span attributes ─────────────────────────────────────
TOOL_NAME = "tenxgraph.tool.name"
TOOL_TYPE = "tenxgraph.tool.type"  # "local" | "mcp"

TOOL_REMOTE = "tenxgraph.tool.remote"  # True when the tool runs on the client

# ── span display names and LangSmith run fields ──────────────────────────────
LOGFIRE_MSG = "logfire.msg"
LANGSMITH_TRACE_NAME = "langsmith.trace.name"
LANGSMITH_SPAN_KIND = "langsmith.span.kind"
LANGSMITH_THREAD_ID = "langsmith.metadata.thread_id"
LANGSMITH_USER_ID = "langsmith.metadata.user_id"
SPAN_UNFINISHED = "tenxgraph.span.unfinished"

# ── lifecycle annotation (added as span event attribute) ─────────────────────
LIFECYCLE = "tenxgraph.lifecycle"

# ── provider name mapping (agentflow provider → OTEL gen_ai.system value) ────
PROVIDER_NAME_MAP: dict[str, str] = {
    "openai": "openai",
    "google": "google_generative_ai",
    "anthropic": "anthropic",
    "azure": "azure",
    "bedrock": "aws.bedrock",
    "cohere": "cohere",
    "mistral": "mistral_ai",
}
