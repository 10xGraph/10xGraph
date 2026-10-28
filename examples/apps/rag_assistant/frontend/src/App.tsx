import { useEffect, useState } from 'react';
import { client } from './client';
import { ChatView } from './components/ChatView';
import { ThreadList } from './components/ThreadList';
import type { ThreadSummary } from './types';

const newId = () => crypto.randomUUID();

export function App() {
  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [active, setActive] = useState<string>(newId);

  const refresh = async () => {
    try {
      const res = await client.threads({ limit: 50 });
      setThreads(res.data.threads.map((t) => ({ id: t.thread_id, name: t.thread_name || t.thread_id.slice(0, 8) })));
    } catch {
      setThreads([]);
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  const remove = async (id: string) => {
    await client.deleteThread(id);
    if (id === active) setActive(newId());
    await refresh();
  };

  return (
    <div className="layout">
      <ThreadList threads={threads} active={active} onSelect={setActive} onNew={() => setActive(newId())} onDelete={remove} />
      <ChatView key={active} threadId={active} onFinished={refresh} />
    </div>
  );
}
