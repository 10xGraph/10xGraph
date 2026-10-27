import { useEffect, useRef, useState } from 'react';
import { useChat } from '../useChat';
import { Sources } from './Sources';

interface Props {
  threadId: string;
  onFinished: () => void;
}

export function ChatView({ threadId, onFinished }: Props) {
  const { messages, busy, error, send, load } = useChat(threadId);
  const [input, setInput] = useState('');
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || busy) return;
    setInput('');
    await send(text);
    onFinished(); // the new thread now exists on the server, so refresh the list
  };

  return (
    <main className="chat">
      <div className="messages">
        {messages.length === 0 && <p className="hint">Ask something about the Lumen docs, for example: what does the Team plan cost?</p>}
        {messages.map((m) => (
          <div key={m.id} className={`bubble ${m.role}`}>
            <p>{m.text || (busy ? '...' : '')}</p>
            {m.role === 'assistant' && <Sources sources={m.sources} />}
          </div>
        ))}
        {error && <p className="error">{error}</p>}
        <div ref={bottom} />
      </div>
      <form onSubmit={submit}>
        <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask the docs..." disabled={busy} />
        <button className="primary" disabled={busy || !input.trim()}>
          Send
        </button>
      </form>
    </main>
  );
}
