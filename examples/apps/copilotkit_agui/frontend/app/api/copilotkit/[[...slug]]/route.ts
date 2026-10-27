// CopilotKit runtime. It runs on the Next.js server, so the 10xGraph API token (if any)
// never reaches the browser.
import { HttpAgent } from "@ag-ui/client";
import {
  CopilotRuntime,
  createCopilotRuntimeHandler,
  InMemoryAgentRunner,
} from "@copilotkit/runtime/v2";

const GRAPH_URL = process.env.GRAPH_API_URL || "http://localhost:8000";

const runtime = new CopilotRuntime({
  agents: {
    // The key ("graph") is the agentId the React side refers to.
    graph: new HttpAgent({
      url: `${GRAPH_URL}/v1/ag-ui`,
      headers: process.env.GRAPH_API_TOKEN
        ? { authorization: `Bearer ${process.env.GRAPH_API_TOKEN}` }
        : {},
    }),
  },
  runner: new InMemoryAgentRunner(),
});

const handler = createCopilotRuntimeHandler({
  runtime,
  basePath: "/api/copilotkit",
});

export const GET = handler;
export const POST = handler;
export const PATCH = handler;
export const DELETE = handler;
