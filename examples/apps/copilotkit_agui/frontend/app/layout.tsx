import type { ReactNode } from "react";

export const metadata = { title: "10xGraph + CopilotKit task board" };

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body style={{ margin: 0, fontFamily: "system-ui, sans-serif" }}>{children}</body>
    </html>
  );
}
