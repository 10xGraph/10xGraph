import type { Source } from '../types';

// Group passages by file so each cited document shows once.
export function Sources({ sources }: { sources: Source[] }) {
  if (sources.length === 0) return null;
  const byFile = new Map<string, Source[]>();
  for (const s of sources) byFile.set(s.source, [...(byFile.get(s.source) ?? []), s]);

  return (
    <details className="sources">
      <summary>Sources ({byFile.size})</summary>
      <ul>
        {[...byFile.entries()].map(([file, hits]) => (
          <li key={file}>
            <strong>{file}</strong>
            {hits.map((h, i) => (
              <p key={i} title={h.content}>
                {h.title || 'section'} (score {h.score})
              </p>
            ))}
          </li>
        ))}
      </ul>
    </details>
  );
}
