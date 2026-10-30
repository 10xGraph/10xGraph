---
title: LLM Providers in 10xGraph - OpenAI, Gemini, Claude, DeepSeek, Ollama
description: Run one 10xGraph ReactAgent on OpenAI, Google Gemini and Vertex AI, Anthropic Claude (direct, Vertex, Bedrock), DeepSeek and Ollama by changing the model string.
section: Examples
group: "Tools and integrations"
order: 60
label: Providers
updated: "2026-10-28"
faq:
  - question: How does 10xGraph know which provider to use?
    answer: It detects the provider from the model string. gpt-* goes to OpenAI, gemini-* to Google, claude-* and anthropic.* to Anthropic. An unknown model name uses the OpenAI client, which is how DeepSeek and Ollama work with base_url.
  - question: How do I use Claude on Vertex AI or Amazon Bedrock?
    answer: Set anthropic_backend="vertex" or anthropic_backend="bedrock" on the agent. Bedrock model ids keep the anthropic. prefix, for example anthropic.claude-sonnet-5-5. Region, project and AWS credentials come from the environment.
  - question: How do I use Gemini through Vertex AI?
    answer: Pass use_vertex_ai=True and set GOOGLE_CLOUD_PROJECT, and optionally GOOGLE_CLOUD_LOCATION. Authentication uses application default credentials instead of an API key.
  - question: Do reasoning options work the same on every provider?
    answer: reasoning_config is a single dict. effort applies to OpenAI, Google and Anthropic, summary is OpenAI only, and thinking_budget is Google only and overrides effort.
---

## What you will build

The same weather-tool ReactAgent running on nine provider setups, chosen by a command-line argument. You also get a reasoning example and a node that calls a provider SDK directly.

## Key concepts

- Model string: `gpt-5-mini`, `gemini-2.5-flash`, `claude-sonnet-5-5`. The provider is detected from it; do not pass `provider=`.
- `api_style`: `"responses"` (the OpenAI Responses API, the default) or `"chat"` (Chat Completions, the default when `base_url` is set).
- `use_vertex_ai=True`: Gemini on Vertex AI.
- `anthropic_backend="vertex"` or `"bedrock"`: Claude on Vertex AI or Amazon Bedrock.
- `base_url` and `api_key`: any OpenAI-compatible server, such as DeepSeek or Ollama.
- `reasoning_config`: `effort`, `summary` (OpenAI), `thinking_budget` (Google).
- `GoogleGenAIConverter`, `OpenAIConverter`: turn raw SDK responses into a `Message`.

## Prerequisites

Install only the extras you need:

```bash
pip install "10xgraph[openai]"            # OpenAI, DeepSeek, Ollama
pip install "10xgraph[google-genai]"      # Gemini and Vertex AI
pip install "10xgraph[anthropic]"         # Claude (also anthropic-vertex, anthropic-bedrock)
```

| Config | Model | Environment |
|---|---|---|
| `openai`, `openai-responses` | `gpt-5-mini` | `OPENAI_API_KEY` |
| `gemini` | `gemini-2.5-flash` | `GEMINI_API_KEY` |
| `gemini-vertex` | `gemini-2.5-flash` | `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, Google credentials |
| `claude` | `claude-sonnet-5-5` | `ANTHROPIC_API_KEY` |
| `claude-vertex` | `claude-sonnet-5-5` | Google credentials, `ANTHROPIC_VERTEX_PROJECT_ID`, `CLOUD_ML_REGION` |
| `claude-bedrock` | `anthropic.claude-sonnet-5-5` | AWS credentials and region |
| `deepseek` | `deepseek-chat` | `DEEPSEEK_API_KEY` |
| `ollama` | `llama3.1` | Ollama running on `localhost:11434` |

## Run it

```bash
python examples/providers/providers.py gemini
python examples/providers/providers.py claude
python examples/providers/reasoning.py openai
python examples/providers/custom_converter.py gemini
```

## How it works

`providers.py` keeps one dict of ReactAgent arguments per provider, with the model visible in each entry:

```python
"openai-responses": {"model": "gpt-5-mini", "api_style": "responses"},
"gemini-vertex": {"model": "gemini-2.5-flash", "use_vertex_ai": True},
"claude-bedrock": {"model": "anthropic.claude-sonnet-5-5", "anthropic_backend": "bedrock"},
"deepseek": {"model": "deepseek-chat", "base_url": "https://api.deepseek.com", ...},
```

The selected entry is unpacked into `ReactAgent(**CONFIGS[name], tools=[get_weather], ...)`. DeepSeek and Ollama set `reasoning_config=None` because those models do not accept OpenAI reasoning options.

`reasoning.py` sets a different `reasoning_config` per provider and then looks for `ReasoningBlock` entries in the reply:

```python
for block in reply.content:
    if isinstance(block, ReasoningBlock):
        print("reasoning:", block.summary[:300])
```

For Google, `effort` maps to a thinking budget (low 512, medium 8192, high 24576) and `thinking_budget` sets it directly. For current Claude models, `effort` selects thinking depth and a token budget is ignored.

`custom_converter.py` is for cases where you want the SDK's own features. A graph node calls `client.aio.models.generate_content(...)` or `client.chat.completions.create(...)` and returns `await GoogleGenAIConverter().convert_response(response)`, so tool calls, text and token usage end up in the same `Message` shape as the Agent class produces.

## Expected output

Illustrative:

```text
provider config: gemini -> model gemini-2.5-flash
assistant: It is sunny and 22C in Paris.
```

## Using a different model

Every entry is just a model string plus options. To try another company, run another config name and install its extra.

## Next steps

- [`../mcp/`](../mcp/) for tools served over MCP.
- [`../dependency_injection/`](../dependency_injection/) for passing your own services to tools.
- [`../quickstart/`](../quickstart/) for the ReAct loop itself.
