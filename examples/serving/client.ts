// Call the served graph from TypeScript with the 10xgraph-client npm package.
// Install: npm install 10xgraph-client
// Run (Node 20+): npx tsx client.ts   (with `10xgraph dev --no-open` running on :8000)
import { TenxGraphClient, Message } from '10xgraph-client';

const client = new TenxGraphClient({
  baseUrl: 'http://localhost:8000',
  // authToken: process.env.TENXGRAPH_TOKEN, // only when "auth" is set in 10xgraph.json
});

// One request, one answer. Reuse the same thread_id to continue a conversation.
const result = await client.invoke([Message.text_message('Weather in Lisbon?', 'user')], {
  config: { thread_id: 'ts-demo' },
});
console.log(result.messages);

// Same call, streamed. Each chunk is one line of the server's /v1/graph/stream response.
const stream = client.stream([Message.text_message('And the local time there?', 'user')], {
  config: { thread_id: 'ts-demo' },
});
for await (const chunk of stream) {
  if (chunk.event === 'message') {
    console.log(chunk.message?.content);
  }
}
