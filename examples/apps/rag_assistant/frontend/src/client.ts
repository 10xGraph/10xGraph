import { TenxGraphClient } from '10xgraph-client';

// Same-origin /api is proxied to the API server (nginx in Docker, Vite in dev).
// If you enable JWT auth in 10xgraph.json, pass `authToken` here.
export const client = new TenxGraphClient({
  baseUrl: `${window.location.origin}/api`,
});
