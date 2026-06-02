"use client";

import { useState, useRef, useCallback, useEffect } from "react";

// ── Types ──────────────────────────────────────────────────────────────────

interface Clause {
  clause_id: string;
  clause_type: string;
  original_text: string;
  is_ambiguous: boolean;
  ambiguity_note: string | null;
  severity_score: number;
  risk_level: "RED" | "YELLOW" | "GREEN";
  risk_label: "HIGH" | "MEDIUM" | "LOW";
  risk_category: string;
  benchmark_comparison: string;
  is_predatory: boolean;
  plain_language_explanation: string;
  scenario_consequence: string;
  key_implications: string[];
  recommended_action: "accept" | "negotiate" | "reject";
  pushback_rationale: string | null;
  alternative_wording: string | null;
  negotiation_tips: string[];
}

interface Report {
  clauses: Clause[];
  overall_score: number;
  red_count: number;
  yellow_count: number;
  green_count: number;
  document_type: string;
  executive_summary: string;
  top_risks: string[];
}

interface AnalyzeResponse {
  filename: string;
  parse_method: string;
  report: Report;
  agents_completed: string[];
}

interface HistoryItem {
  id: string;
  filename: string;
  created_at: string;
  doc_type: string;
  score: number;
  red_count: number;
  yellow_count: number;
  green_count: number;
}

// ── Constants ──────────────────────────────────────────────────────────────

const RISK_COLORS = {
  RED:    { bg: "#3d1a1a", border: "#ef5350", badge: "#ef5350", text: "#ffcdd2" },
  YELLOW: { bg: "#2d2a10", border: "#ffc107", badge: "#ffc107", text: "#fff8e1" },
  GREEN:  { bg: "#1a2e1a", border: "#66bb6a", badge: "#66bb6a", text: "#e8f5e9" },
};

const ACTION_COLORS: Record<string, string> = {
  reject: "#ef5350", negotiate: "#ffc107", accept: "#66bb6a",
};

// ── Client ID — anonymous, per-browser, persisted in localStorage ──────────

function getClientId(): string {
  if (typeof window === "undefined") return "ssr";
  let id = localStorage.getItem("lexguard_client_id");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("lexguard_client_id", id);
  }
  return id;
}

// ── Main page ──────────────────────────────────────────────────────────────

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [isPrinting, setIsPrinting] = useState(false);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // ── Expand all clauses before print, restore after ────────────────────
  useEffect(() => {
    const onBefore = () => setIsPrinting(true);
    const onAfter  = () => setIsPrinting(false);
    window.addEventListener("beforeprint", onBefore);
    window.addEventListener("afterprint",  onAfter);
    return () => {
      window.removeEventListener("beforeprint", onBefore);
      window.removeEventListener("afterprint",  onAfter);
    };
  }, []);

  // ── Load history on mount ──────────────────────────────────────────────

  const loadHistory = useCallback(async () => {
    setHistoryLoading(true);
    try {
      const res = await fetch("/api/history", {
        headers: { "X-Client-ID": getClientId() },
      });
      if (res.ok) setHistory(await res.json());
    } catch { /* history is optional */ }
    finally { setHistoryLoading(false); }
  }, []);

  useEffect(() => { loadHistory(); }, [loadHistory]);

  // ── File handling ────────────────────────────────────────────────────────

  const validateFile = (f: File): string | null => {
    const ext = f.name.split(".").pop()?.toLowerCase();
    if (!["pdf", "docx", "doc"].includes(ext ?? "")) return "Only PDF and DOCX files are supported.";
    if (f.size > 15 * 1024 * 1024) return "File must be under 15 MB.";
    if (f.size === 0) return "File is empty.";
    return null;
  };

  const handleFile = (f: File) => {
    const err = validateFile(f);
    if (err) { setError(err); return; }
    setFile(f); setError(null); setResult(null);
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer.files[0]; if (f) handleFile(f);
  }, []);
  const onDragOver = useCallback((e: React.DragEvent) => { e.preventDefault(); setDragging(true); }, []);
  const onDragLeave = useCallback(() => setDragging(false), []);

  // ── Analyze ──────────────────────────────────────────────────────────────

  const analyze = async () => {
    if (!file) return;
    setLoading(true); setError(null); setResult(null);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
        headers: { "X-Client-ID": getClientId() },
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `Server error ${res.status}`);
      }
      const data: AnalyzeResponse = await res.json();
      setResult(data);
      loadHistory(); // refresh sidebar
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unexpected error — check the browser console.");
    } finally { setLoading(false); }
  };

  const reset = () => { setFile(null); setResult(null); setError(null); };

  // ── Load a history item ───────────────────────────────────────────────

  const loadHistoryItem = async (id: string) => {
    try {
      const res = await fetch(`/api/history/${id}`, {
        headers: { "X-Client-ID": getClientId() },
      });
      if (!res.ok) return;
      const data: AnalyzeResponse = await res.json();
      setResult(data); setFile(null); setError(null); setShowHistory(false);
    } catch { /* silent */ }
  };

  const deleteHistoryItem = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    await fetch(`/api/history/${id}`, {
      method: "DELETE",
      headers: { "X-Client-ID": getClientId() },
    });
    setHistory(h => h.filter(x => x.id !== id));
  };

  // ── Download report ────────────────────────────────────────────────────

  const downloadReport = () => {
    // beforeprint event sets isPrinting=true which expands all clauses.
    // Small timeout lets React re-render before the print dialog opens.
    setIsPrinting(true);
    setTimeout(() => { window.print(); }, 150);
  };

  // ── Helpers ───────────────────────────────────────────────────────────

  const riskColor = (level: "RED" | "YELLOW" | "GREEN") => RISK_COLORS[level];
  const overallColor = result
    ? result.report.overall_score >= 7 ? "#ef5350"
    : result.report.overall_score >= 4 ? "#ffc107" : "#66bb6a"
    : "#888";

  const fmtDate = (iso: string) =>
    new Date(iso).toLocaleDateString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });

  // ── Render ────────────────────────────────────────────────────────────

  return (
    <>
      {/* ── Styles ── */}
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @media (max-width: 640px) { .grid-2 { grid-template-columns: 1fr !important; } }

        @media print {
          /* Hide interactive chrome */
          .no-print, button, input, label[for="contract-file-input"] { display: none !important; }

          /* Reset to readable black-on-white */
          body, main { background: #fff !important; color: #111 !important;
            font-family: Georgia, serif; font-size: 11pt; }
          main { max-width: 100% !important; padding: 0.4in !important; }

          /* Page header */
          h1 { color: #1e1b4b !important; font-size: 18pt !important; }
          h2 { font-size: 13pt !important; color: #111 !important; margin-top: 18pt; }

          /* Score card */
          div[style*="text-align: center"] { border: 2px solid #333 !important;
            background: #fff !important; padding: 12pt !important; margin-bottom: 14pt; }

          /* Clause cards — force visible, avoid splitting across pages */
          article { border: 1px solid #aaa !important; background: #fff !important;
            break-inside: avoid; page-break-inside: avoid;
            margin-bottom: 10pt; padding: 8pt 10pt; }

          /* Clause header button text — keep visible */
          article button { display: block !important; font-weight: bold !important;
            font-size: 10pt !important; color: #111 !important;
            background: none !important; border: none !important;
            padding: 0 !important; text-align: left !important; width: 100%; }

          /* Detail sections */
          article div[style*="padding: \"0 18px 18px\""] { display: block !important; }
          blockquote { border-left: 3px solid #666 !important; background: #f5f5f5 !important;
            color: #333 !important; padding: 6pt 10pt !important; font-size: 9pt !important; }

          /* Info blocks */
          div[style*="background: \"#0f1117\""] { background: #f0f0f0 !important;
            border: 1px solid #ccc !important; color: #111 !important; }

          /* Alternative wording */
          div[style*="background: \"#0f2414\""] { background: #edfaed !important;
            border: 1px solid #4caf50 !important; color: #111 !important; }

          /* Risk badges — keep colour as background but use dark text */
          span[style*="color: \"#000\""] { -webkit-print-color-adjust: exact;
            print-color-adjust: exact; }

          /* Footer */
          footer { border: 1px solid #ccc !important; background: #fff !important;
            color: #555 !important; font-size: 8pt !important; }

          /* Force collapse arrows hidden in print */
          span[style*="font-size: 11"] { display: none !important; }
        }
      `}</style>

      <main style={{ maxWidth: 920, margin: "0 auto", padding: "28px 16px", position: "relative" }}>

        {/* ── Header ── */}
        <header style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 36 }}>
          <div>
            <h1 style={{ fontSize: 30, fontWeight: 800, color: "#7c8cf8", margin: 0 }}>⚖ LEXGUARD</h1>
            <p style={{ color: "#6b7280", fontSize: 13, margin: "4px 0 0" }}>
              AI contract risk analysis — understand what you&apos;re signing
            </p>
          </div>
          <button
            className="no-print"
            onClick={() => { setShowHistory(!showHistory); loadHistory(); }}
            aria-label="Toggle history"
            style={{ background: showHistory ? "#312e81" : "#1f2937", border: "1px solid #374151",
              color: "#e8eaf6", padding: "8px 14px", borderRadius: 8, cursor: "pointer", fontSize: 13,
              display: "flex", alignItems: "center", gap: 6 }}
          >
            📋 History {history.length > 0 && (
              <span style={{ background: "#7c8cf8", color: "#fff", borderRadius: 10,
                padding: "1px 7px", fontSize: 11, fontWeight: 700 }}>
                {history.length}
              </span>
            )}
          </button>
        </header>

        {/* ── History panel ── */}
        {showHistory && (
          <aside className="no-print" style={{ background: "#161b27", border: "1px solid #374151",
            borderRadius: 12, padding: 16, marginBottom: 24 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
              <span style={{ color: "#e8eaf6", fontWeight: 700, fontSize: 15 }}>Past analyses</span>
              <button onClick={() => setShowHistory(false)}
                style={{ background: "none", border: "none", color: "#6b7280", cursor: "pointer", fontSize: 18 }}>✕</button>
            </div>
            {historyLoading && <p style={{ color: "#6b7280", fontSize: 13 }}>Loading…</p>}
            {!historyLoading && history.length === 0 && (
              <p style={{ color: "#6b7280", fontSize: 13 }}>No analyses yet. Upload a contract to get started.</p>
            )}
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {history.map(item => (
                <div key={item.id} onClick={() => loadHistoryItem(item.id)}
                  style={{ background: "#0f1117", border: "1px solid #374151", borderRadius: 8,
                    padding: "10px 14px", cursor: "pointer", display: "flex",
                    justifyContent: "space-between", alignItems: "center" }}
                  role="button" tabIndex={0}
                  onKeyDown={e => e.key === "Enter" && loadHistoryItem(item.id)}>
                  <div>
                    <div style={{ color: "#e8eaf6", fontSize: 13, fontWeight: 600 }}>{item.filename}</div>
                    <div style={{ color: "#6b7280", fontSize: 11, marginTop: 2 }}>
                      {fmtDate(item.created_at)} · {item.doc_type.replace(/_/g, " ")} ·{" "}
                      <span style={{ color: item.score >= 7 ? "#ef5350" : item.score >= 4 ? "#ffc107" : "#66bb6a", fontWeight: 700 }}>
                        {item.score.toFixed(1)}/10
                      </span>
                      {" "}· R={item.red_count} Y={item.yellow_count} G={item.green_count}
                    </div>
                  </div>
                  <button onClick={e => deleteHistoryItem(item.id, e)}
                    aria-label="Delete"
                    style={{ background: "none", border: "none", color: "#6b7280", cursor: "pointer",
                      fontSize: 16, padding: "2px 6px", borderRadius: 4 }}>✕</button>
                </div>
              ))}
            </div>
          </aside>
        )}

        {/* ── Upload zone ── */}
        {!result && (
          <section aria-label="Document upload" className="no-print">
            <label htmlFor="contract-file-input"
              onDrop={onDrop} onDragOver={onDragOver} onDragLeave={onDragLeave}
              style={{ display: "block", border: `2px dashed ${dragging ? "#7c8cf8" : file ? "#66bb6a" : "#4f46e5"}`,
                borderRadius: 16, padding: "40px 24px", textAlign: "center", cursor: "pointer",
                background: dragging ? "#1a1d2e" : file ? "#0d1f0d" : "#161b27", transition: "all 0.2s", marginBottom: 16 }}>
              <input id="contract-file-input" ref={inputRef} type="file" accept=".pdf,.docx,.doc"
                style={{ display: "none" }}
                onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])}
                aria-label="Select PDF or DOCX file" />
              {file ? (
                <>
                  <div style={{ fontSize: 38, marginBottom: 8 }}>✅</div>
                  <p style={{ color: "#66bb6a", fontWeight: 700, fontSize: 17, margin: "0 0 4px" }}>{file.name}</p>
                  <p style={{ color: "#9ca3af", fontSize: 12, margin: "0 0 12px" }}>
                    {(file.size / 1024).toFixed(1)} KB · click to change
                  </p>
                </>
              ) : (
                <>
                  <div style={{ fontSize: 38, marginBottom: 12 }}>📂</div>
                  <p style={{ color: "#e8eaf6", fontWeight: 700, fontSize: 17, margin: "0 0 6px" }}>
                    Click to upload or drag your contract here
                  </p>
                  <p style={{ color: "#6b7280", fontSize: 12, margin: "0 0 16px" }}>PDF or DOCX · max 15 MB</p>
                  <span style={{ display: "inline-block", background: "#4f46e5", color: "#fff",
                    padding: "9px 22px", borderRadius: 8, fontWeight: 600, fontSize: 13, pointerEvents: "none" }}>
                    Browse files
                  </span>
                </>
              )}
            </label>

            {error && (
              <div role="alert" style={{ background: "#3d1a1a", border: "1px solid #ef5350",
                borderRadius: 8, padding: "11px 16px", color: "#ffcdd2", marginBottom: 14, fontSize: 13 }}>
                ⚠ {error}
              </div>
            )}

            <button onClick={file && !loading ? analyze : undefined}
              disabled={!file || loading} aria-label={file ? "Analyze contract" : "Select a file first"}
              style={{ width: "100%", padding: "15px", borderRadius: 12, border: "none",
                background: file && !loading ? "#7c8cf8" : "#1f2937",
                color: file && !loading ? "#fff" : "#4b5563",
                fontSize: 15, fontWeight: 700,
                cursor: file && !loading ? "pointer" : "default", transition: "all 0.2s" }}>
              {loading ? "⏳ Analyzing with 4 AI agents…"
                : file ? "→ Analyze Contract"
                : "↑ Select a file above to begin"}
            </button>

            {loading && (
              <>
                {/* Loading skeleton */}
                <div style={{ marginTop: 20 }}>
                  {["Extracting clauses…", "Scoring risks…", "Explaining in plain language…", "Writing negotiation advice…"]
                    .map((label, i) => (
                      <div key={i} style={{ display: "flex", alignItems: "center", gap: 12,
                        padding: "10px 14px", background: "#161b27", borderRadius: 8, marginBottom: 6,
                        border: "1px solid #374151" }}>
                        <div style={{ width: 18, height: 18, borderRadius: "50%",
                          border: "2px solid #7c8cf8", borderTopColor: "transparent",
                          animation: "spin 1s linear infinite" }} />
                        <span style={{ color: "#9ca3af", fontSize: 13 }}>Agent {i + 1}: {label}</span>
                      </div>
                    ))}
                </div>
                <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
                <p role="status" aria-live="polite"
                  style={{ textAlign: "center", color: "#6b7280", marginTop: 12, fontSize: 12 }}>
                  Running 4-agent AI pipeline · 60–120 seconds for a full contract
                </p>
              </>
            )}
          </section>
        )}

        {/* ── Results ── */}
        {result && (
          <section aria-label="Analysis results">

            {/* Overall score card */}
            <div style={{ background: "#161b27", border: `2px solid ${overallColor}`,
              borderRadius: 16, padding: "22px", marginBottom: 22, textAlign: "center" }}>
              <p style={{ color: "#9ca3af", margin: "0 0 6px", fontSize: 13 }}>
                {result.filename} — {result.report.document_type.replace(/_/g, " ").toUpperCase()}
              </p>
              <div style={{ display: "flex", justifyContent: "center", alignItems: "baseline", gap: 8, marginBottom: 12 }}>
                <span aria-label={`Overall risk score ${result.report.overall_score} out of 10`}
                  style={{ fontSize: 52, fontWeight: 900, color: overallColor, lineHeight: 1 }}>
                  {result.report.overall_score.toFixed(1)}
                </span>
                <span style={{ color: "#9ca3af", fontSize: 16 }}>/10</span>
                <span style={{ background: overallColor, color: "#000", fontWeight: 800,
                  padding: "3px 12px", borderRadius: 20, fontSize: 13 }}>
                  {result.report.overall_score >= 7 ? "HIGH RISK" : result.report.overall_score >= 4 ? "MEDIUM RISK" : "LOW RISK"}
                </span>
              </div>

              <div style={{ display: "flex", justifyContent: "center", gap: 10, marginBottom: 14 }}
                role="list" aria-label="Risk level summary">
                {[
                  { count: result.report.red_count,    label: "HIGH",   color: "#ef5350" },
                  { count: result.report.yellow_count, label: "MEDIUM", color: "#ffc107" },
                  { count: result.report.green_count,  label: "LOW",    color: "#66bb6a" },
                ].map(({ count, label, color }) => (
                  <div key={label} role="listitem"
                    aria-label={`${count} ${label} risk clauses`}
                    style={{ background: "#0f1117", border: `1px solid ${color}`,
                      borderRadius: 8, padding: "7px 14px", textAlign: "center" }}>
                    <div style={{ fontSize: 22, fontWeight: 900, color }}>{count}</div>
                    <div style={{ fontSize: 10, color: "#9ca3af" }}>{label} RISK</div>
                  </div>
                ))}
              </div>

              <p style={{ color: "#d1d5db", fontSize: 13, margin: "0 0 14px", lineHeight: 1.6, textAlign: "left" }}>
                {result.report.executive_summary}
              </p>

              <div className="no-print" style={{ display: "flex", gap: 8, justifyContent: "center", flexWrap: "wrap" }}>
                <button onClick={reset}
                  style={{ padding: "8px 18px", borderRadius: 8, border: "1px solid #374151",
                    background: "transparent", color: "#9ca3af", cursor: "pointer", fontSize: 13 }}>
                  ↩ Analyze another
                </button>
                <button onClick={downloadReport}
                  style={{ padding: "8px 18px", borderRadius: 8, border: "1px solid #4f46e5",
                    background: "#1e1b4b", color: "#a5b4fc", cursor: "pointer", fontSize: 13 }}>
                  ⬇ Download PDF report
                </button>
              </div>
            </div>

            {/* ── On-the-Go TL;DR ── */}
            <div style={{ background: "#0d1117", border: "1px solid #374151", borderRadius: 16, padding: "24px", marginBottom: 32 }}>
              <h2 style={{ fontSize: 20, fontWeight: 800, color: "#7c8cf8", margin: "0 0 16px", display: "flex", alignItems: "center", gap: 8 }}>
                ⚡ On-the-Go Summary
              </h2>
              
              {result.report.top_risks.length > 0 && (
                <div style={{ marginBottom: 24 }}>
                  <h3 style={{ fontSize: 13, fontWeight: 700, color: "#9ca3af", textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 12 }}>
                    🚨 Top Risks
                  </h3>
                  <ul style={{ margin: 0, paddingLeft: 20, color: "#fca5a5", fontSize: 14, lineHeight: 1.6 }}>
                    {result.report.top_risks.map((risk, i) => (
                      <li key={i} style={{ marginBottom: 6 }}>{risk}</li>
                    ))}
                  </ul>
                </div>
              )}

              <h3 style={{ fontSize: 13, fontWeight: 700, color: "#9ca3af", textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 12 }}>
                🛑 Actionable Points (Read carefully)
              </h3>
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                {result.report.clauses
                  .filter(c => c.risk_level === "RED" || c.risk_level === "YELLOW")
                  .map(clause => {
                    const c = riskColor(clause.risk_level);
                    return (
                      <div key={clause.clause_id} style={{ background: "#161b27", borderLeft: `4px solid ${c.border}`, borderRadius: "0 8px 8px 0", padding: "12px 16px" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                          <span style={{ background: c.badge, color: "#000", fontWeight: 800, padding: "2px 8px", borderRadius: 4, fontSize: 11 }}>
                            {clause.risk_label}
                          </span>
                          <span style={{ color: "#9ca3af", fontSize: 12, fontWeight: 600 }}>{clause.clause_type.replace(/_/g, " ")}</span>
                        </div>
                        <p style={{ color: "#e8eaf6", fontSize: 14, margin: "0 0 8px", lineHeight: 1.5 }}>
                          {clause.plain_language_explanation}
                        </p>
                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                          <span style={{ color: "#9ca3af", fontSize: 12 }}>Recommendation:</span>
                          <span style={{ background: ACTION_COLORS[clause.recommended_action], color: "#000", fontWeight: 700, padding: "2px 10px", borderRadius: 5, fontSize: 11, textTransform: "uppercase" }}>
                            {clause.recommended_action}
                          </span>
                        </div>
                      </div>
                    );
                })}
                {result.report.clauses.filter(c => c.risk_level === "RED" || c.risk_level === "YELLOW").length === 0 && (
                  <p style={{ color: "#66bb6a", fontSize: 14, margin: 0 }}>No high or medium risk clauses detected. You are good to go!</p>
                )}
              </div>
            </div>

            {/* Clause cards */}
            <h2 style={{ fontSize: 16, fontWeight: 700, color: "#e8eaf6", margin: "0 0 14px" }}>
              Detailed Clause-by-clause analysis ({result.report.clauses.length} clauses)
            </h2>

            <div role="list" aria-label="Contract clauses">
              {result.report.clauses.map(clause => {
                const c = riskColor(clause.risk_level);
                const expanded = isPrinting || expandedId === clause.clause_id;
                return (
                  <article key={clause.clause_id} role="listitem"
                    aria-label={`${clause.clause_id}: ${clause.risk_label} risk, score ${clause.severity_score}`}
                    style={{ background: c.bg, border: `1px solid ${c.border}`,
                      borderRadius: 12, marginBottom: 10, overflow: "hidden" }}>

                    <button onClick={() => setExpandedId(expanded ? null : clause.clause_id)}
                      aria-expanded={expanded}
                      style={{ width: "100%", background: "none", border: "none",
                        padding: "14px 18px", cursor: "pointer", textAlign: "left",
                        display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
                      <span aria-label={`${clause.risk_label} severity ${clause.severity_score}`}
                        style={{ background: c.badge, color: "#000", fontWeight: 800,
                          padding: "3px 9px", borderRadius: 5, fontSize: 11, whiteSpace: "nowrap" }}>
                        {clause.risk_level} · {clause.severity_score.toFixed(1)} · {clause.risk_label}
                      </span>
                      <span style={{ background: "#1f2937", color: "#9ca3af",
                        padding: "2px 7px", borderRadius: 4, fontSize: 10, whiteSpace: "nowrap" }}>
                        {clause.clause_type.replace(/_/g, " ")}
                      </span>
                      {clause.is_predatory && (
                        <span style={{ background: "#7f1d1d", color: "#fca5a5",
                          padding: "2px 7px", borderRadius: 4, fontSize: 10 }}>⚠ PREDATORY</span>
                      )}
                      <span style={{ color: c.text, fontSize: 13, flexGrow: 1,
                        overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        {clause.original_text.slice(0, 100)}{clause.original_text.length > 100 ? "…" : ""}
                      </span>
                      <span style={{ color: "#6b7280", fontSize: 11, flexShrink: 0 }}>{expanded ? "▲" : "▼"}</span>
                    </button>

                    {expanded && (
                      <div style={{ padding: "0 18px 18px", borderTop: `1px solid ${c.border}` }}>
                        <blockquote style={{ background: "#0f1117", borderLeft: `3px solid ${c.border}`,
                          margin: "14px 0", padding: "10px 14px", color: "#9ca3af",
                          fontSize: 12, fontStyle: "italic", lineHeight: 1.6 }}>
                          &ldquo;{clause.original_text}&rdquo;
                        </blockquote>

                        {clause.is_ambiguous && clause.ambiguity_note && (
                          <div style={{ background: "#1f1a00", border: "1px solid #ffc107",
                            borderRadius: 7, padding: "9px 13px", marginBottom: 10,
                            fontSize: 12, color: "#fff8e1" }}>
                            <strong>⚡ Ambiguity:</strong> {clause.ambiguity_note}
                          </div>
                        )}

                        <div className="grid-2" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 10 }}>
                          <InfoBlock label="Plain English" text={clause.plain_language_explanation} color={c.text} />
                          <InfoBlock label="If you sign this…" text={clause.scenario_consequence} color={c.text} />
                        </div>

                        <InfoBlock label="Benchmark comparison" text={clause.benchmark_comparison} color="#d1d5db" />

                        <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 10 }}>
                          <span style={{ color: "#9ca3af", fontSize: 12 }}>Recommended action:</span>
                          <span style={{ background: ACTION_COLORS[clause.recommended_action],
                            color: "#000", fontWeight: 700, padding: "2px 10px",
                            borderRadius: 5, fontSize: 11, textTransform: "uppercase" }}>
                            {clause.recommended_action}
                          </span>
                        </div>

                        {clause.pushback_rationale && (
                          <p style={{ color: "#d1d5db", fontSize: 12, margin: "8px 0", lineHeight: 1.6 }}>
                            <strong style={{ color: "#9ca3af" }}>Why push back:</strong> {clause.pushback_rationale}
                          </p>
                        )}

                        {clause.alternative_wording && (
                          <div style={{ background: "#0f2414", border: "1px solid #66bb6a",
                            borderRadius: 7, padding: "9px 13px", fontSize: 12,
                            color: "#e8f5e9", lineHeight: 1.6, marginTop: 8 }}>
                            <strong>✍ Suggested wording:</strong><br />
                            &ldquo;{clause.alternative_wording}&rdquo;
                          </div>
                        )}

                        {clause.negotiation_tips.length > 0 && (
                          <ul style={{ color: "#d1d5db", fontSize: 12, margin: "10px 0 0", paddingLeft: 18 }}>
                            {clause.negotiation_tips.map((tip, i) => (
                              <li key={i} style={{ marginBottom: 3 }}>{tip}</li>
                            ))}
                          </ul>
                        )}
                      </div>
                    )}
                  </article>
                );
              })}
            </div>
          </section>
        )}

        {/* ── Disclaimer — always visible ── */}
        <footer role="contentinfo" aria-label="Legal disclaimer"
          style={{ marginTop: 36, padding: "14px 18px", background: "#161b27",
            border: "1px solid #374151", borderRadius: 12,
            textAlign: "center", color: "#6b7280", fontSize: 11, lineHeight: 1.6 }}>
          <strong style={{ color: "#9ca3af" }}>⚖ Legal Disclaimer</strong><br />
          LEXGUARD is an AI tool for informational purposes only. It does <strong>not</strong> constitute legal
          advice and is <strong>not</strong> a substitute for a qualified legal professional.
          Always consult a licensed attorney before signing any contract.
        </footer>
      </main>
    </>
  );
}

function InfoBlock({ label, text, color }: { label: string; text: string; color: string }) {
  return (
    <div style={{ background: "#0f1117", borderRadius: 7, padding: "9px 12px" }}>
      <div style={{ color: "#6b7280", fontSize: 10, fontWeight: 700, marginBottom: 3,
        textTransform: "uppercase", letterSpacing: 0.5 }}>{label}</div>
      <div style={{ color, fontSize: 12, lineHeight: 1.6 }}>{text}</div>
    </div>
  );
}
