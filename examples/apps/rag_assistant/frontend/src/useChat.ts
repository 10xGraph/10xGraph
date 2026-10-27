import { useCallback, useRef, useState } from 'react';
import { Message, StreamEventType } from '10xgraph-client';
import { client } from './client';
import type { ChatMessage, Source } from './types';

// The search_docs tool returns a JSON string: [{ source, title, score, content }].
function parseSources(output: unknown): Source[] {
  try {
    const data = typeof output === 'string' ? JSON.parse(output) : output;
    return Array.isArray(data) ? (data as Source[]).filter((s) => s && s.source) : [];
  } catch {
    return [];
  }
}

export function useChat(threadId: string) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abort = useRef(false);

  // Load the saved history of an existing thread (tool results carry the sources).
  const load = useCallback(async () => {
    setMessages([]);
    setError(null);
    try {
      const res = await client.threadMessages({ threadId });
      const out: ChatMessage[] = [];
      let pending: Source[] = [];
      for (const m of res.data.messages) {
        const blocks = (m.content ?? []) as any[];
        if (m.role === 'tool') {
          for (const b of blocks) if (b.type === 'tool_result') pending.push(...parseSources(b.output));
        } else if (m.role === 'user' || m.role === 'assistant') {
          const text = blocks.filter((b) => b.type === 'text').map((b) => b.text).join('');
          if (!text) continue;
          out.push({ id: m.message_id ?? crypto.randomUUID(), role: m.role, text, sources: m.role === 'assistant' ? pending : [] });
          if (m.role === 'assistant') pending = [];
        }
      }
      setMessages(out);
    } catch {
      // A brand new thread has no history yet.
    }
  }, [threadId]);

  const send = useCallback(
    async (text: string) => {
      setError(null);
      setBusy(true);
      abort.current = false;
      const assistantId = crypto.randomUUID();
      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: 'user', text, sources: [] },
        { id: assistantId, role: 'assistant', text: '', sources: [] },
      ]);
      const patch = (fn: (m: ChatMessage) => ChatMessage) =>
        setMessages((prev) => prev.map((m) => (m.id === assistantId ? fn(m) : m)));

      try {
        const stream = client.stream([Message.text_message(text, 'user')], {
          config: { thread_id: threadId },
          response_granularity: 'low',
        });
        for await (const chunk of stream) {
          if (abort.current) break;
          if (chunk.event === StreamEventType.ERROR) {
            setError(String(chunk.data?.message ?? 'The run failed'));
            continue;
          }
          const msg = chunk.message;
          if (chunk.event !== StreamEventType.MESSAGE || !msg) continue;
          const blocks = (msg.content ?? []) as any[];

          if (msg.role === 'tool') {
            const found = blocks.filter((b) => b.type === 'tool_result').flatMap((b) => parseSources(b.output));
            if (found.length) patch((m) => ({ ...m, sources: [...m.sources, ...found] }));
          } else if (msg.role === 'assistant') {
            const t = blocks.filter((b) => b.type === 'text').map((b) => b.text).join('');
            if (!t) continue;
            // Deltas append. The final non-delta message carries the full text of that turn.
            patch((m) => ({ ...m, text: msg.delta ? m.text + t : t }));
          }
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [threadId],
  );

  return { messages, busy, error, send, load };
}
