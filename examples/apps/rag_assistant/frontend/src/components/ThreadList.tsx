import type { ThreadSummary } from '../types';

interface Props {
  threads: ThreadSummary[];
  active: string;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
}

export function ThreadList({ threads, active, onSelect, onNew, onDelete }: Props) {
  return (
    <aside className="sidebar">
      <button className="primary" onClick={onNew}>
        New thread
      </button>
      <ul>
        {threads.map((t) => (
          <li key={t.id} className={t.id === active ? 'active' : ''}>
            <button className="link" onClick={() => onSelect(t.id)}>
              {t.name}
            </button>
            <button className="icon" aria-label="Delete thread" onClick={() => onDelete(t.id)}>
              x
            </button>
          </li>
        ))}
      </ul>
    </aside>
  );
}
