"use client";

import { useState } from "react";
import { CopilotChat, CopilotKitProvider, useFrontendTool } from "@copilotkit/react-core/v2";
import "@copilotkit/react-core/v2/styles.css";
import { z } from "zod";

import { Approval } from "@/components/Approval";
import { Board } from "@/components/Board";

// Frontend tool: the model calls it, the run ends with the call unanswered, this handler runs
// in the browser, and CopilotKit starts a new run with the result.
function FocusTool({ onFocus }: { onFocus: (id: number) => void }) {
  useFrontendTool({
    name: "focus_task",
    description: "Highlight a task on the board in the user's browser.",
    parameters: z.object({ task_id: z.number().describe("The task id to highlight") }),
    handler: async ({ task_id }) => {
      onFocus(task_id);
      return `Highlighted task ${task_id}`;
    },
  });
  return null;
}

export default function Page() {
  const [focusedId, setFocusedId] = useState<number | null>(null);

  return (
    <CopilotKitProvider runtimeUrl="/api/copilotkit" agentId="graph">
      <FocusTool onFocus={setFocusedId} />
      <Approval />
      <div style={{ display: "grid", gridTemplateColumns: "320px 1fr", height: "100vh" }}>
        <aside style={{ borderRight: "1px solid #eee", overflow: "auto" }}>
          <Board focusedId={focusedId} />
        </aside>
        <CopilotChat agentId="graph" labels={{ title: "Task assistant" }} />
      </div>
    </CopilotKitProvider>
  );
}
