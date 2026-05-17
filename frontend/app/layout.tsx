import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "LEXGUARD — AI Contract Intelligence",
  description: "AI-powered contract risk analysis. Upload any legal document to identify hidden risks, predatory clauses, and negotiation opportunities.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ margin: 0, fontFamily: "'Segoe UI', system-ui, sans-serif", background: "#0f1117", color: "#e8eaf6", minHeight: "100vh" }}>
        {children}
      </body>
    </html>
  );
}
