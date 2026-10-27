"use client";

import { useAgent } from "@copilotkit/react-core/v2";

type Task = { id: number; title: string; status: "todo" | "doing" | "done"; priority: string };
type BoardState = { board_title?: string; tasks?: Task[] };

const COLUMNS: Task["status"][] = ["todo", "doing", "done"];

// Reads the graph's BoardState. Each STATE_SNAPSHOT from the server updates agent.state.
export function Board({ focusedId }: { focusedId: number | null }) {
  const { agent } = useAgent({ agentId: "graph" });
  const state = (agent.state ?? {}) as BoardState;
  const tasks = state.tasks ?? [];

  return (
    <div style={{ padding: 16 }}>
      <h2 style={{ marginTop: 0 }}>{state.board_title ?? "My task board"}</h2>
      {COLUMNS.map((status) => (
        <section key={status} style={{ marginBottom: 16 }}>
          <h3 style={{ textTransform: "uppercase", fontSize: 12, color: "#666" }}>{status}</h3>
          {tasks
            .filter((t) => t.status === status)
            .map((t) => (
              <div
                key={t.id}
                style={{
                  padding: 8,
                  marginBottom: 6,
                  borderRadius: 6,
                  border: "1px solid #ddd",
                  background: t.id === focusedId ? "#fff3b0" : "white",
                }}
              >
                #{t.id} {t.title} <small style={{ color: "#888" }}>({t.priority})</small>
              </div>
            ))}
        </section>
      ))}
      {tasks.length === 0 && <p style={{ color: "#888" }}>No tasks yet. Ask the assistant.</p>}
    </div>
  );
}
