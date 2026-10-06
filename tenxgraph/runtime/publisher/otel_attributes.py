"""OTEL attribute name constants aligned with GenAI semantic conventions.

https://opentelemetry.io/docs/specs/semconv/gen-ai/
"""

# ── gen_ai core ──────────────────────────────────────────────────────────────
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_OPERATION = "gen_ai.operation.name"

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

# ── agentflow.graph span attributes ─────────────────────────────────────────
GRAPH_THREAD_ID = "agentflow.graph.thread_id"
GRAPH_RUN_ID = "agentflow.graph.run_id"
GRAPH_USER_ID = "agentflow.graph.user_id"
GRAPH_TOTAL_STEPS = "agentflow.graph.total_steps"
GRAPH_MODEL = "agentflow.graph.model"

# ── agentflow.node span attributes ───────────────────────────────────────────
NODE_NAME = "agentflow.node.name"
NODE_STEP = "agentflow.node.step"

# ── agentflow.tool span attributes ───────────────────────────────────────────
TOOL_NAME = "agentflow.tool.name"
TOOL_TYPE = "agentflow.tool.type"  # "local" | "mcp"

# ── lifecycle annotation (added as span event attribute) ─────────────────────
LIFECYCLE = "agentflow.lifecycle"

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
