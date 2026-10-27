"use client";

import { useInterrupt } from "@copilotkit/react-core/v2";

// Renders the approval requested by interrupt() in the clear_done_tasks tool.
// resolve(payload) resumes the graph and interrupt() returns payload; cancel() returns None.
export function Approval() {
  useInterrupt({
    render: ({ interrupt, resolve, cancel }) => {
      if (!interrupt) return null;
      return (
        <div
          style={{
            position: "fixed",
            bottom: 20,
            left: 20,
            background: "white",
            border: "1px solid #ccc",
            borderRadius: 8,
            padding: 16,
            boxShadow: "0 2px 8px rgba(0,0,0,0.15)",
            maxWidth: 300,
          }}
        >
          <p style={{ marginTop: 0 }}>{interrupt.message}</p>
          <div style={{ display: "flex", gap: 8 }}>
            <button onClick={() => resolve({ approved: true })} style={{ flex: 1 }}>
              Approve
            </button>
            <button onClick={() => resolve({ approved: false })} style={{ flex: 1 }}>
              Reject
            </button>
            <button onClick={() => cancel()}>Cancel</button>
          </div>
        </div>
      );
    },
  });
  return null;
}
