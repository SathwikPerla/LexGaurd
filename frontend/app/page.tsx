"use client";

import { useState, useRef, useCallback } from "react";

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

// ── Color helpers ──────────────────────────────────────────────────────────

const RISK_COLORS = {
  RED: { bg: "#3d1a1a", border: "#ef5350", badge: "#ef5350", text: "#ffcdd2" },
  YELLOW: { bg: "#2d2a10", border: "#ffc107", badge: "#ffc107", text: "#fff8e1" },
  GREEN: { bg: "#1a2e1a", border: "#66bb6a", badge: "#66bb6a", text: "#e8f5e9" },
};

const ACTION_COLORS = {
  reject: "#ef5350",
  negotiate: "#ffc107",
  accept: "#66bb6a",
};

// ── Main page ──────────────────────────────────────────────────────────────

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // ── File handling ────────────────────────────────────────────────────────

  const validateFile = (f: File): string | null => {
    const ext = f.name.split(".").pop()?.toLowerCase();
    if (!["pdf", "docx", "doc"].includes(ext ?? ""))
      return "Only PDF and DOCX files are supported.";
    if (f.size > 15 * 1024 * 1024)
      return "File must be under 15 MB.";
    if (f.size === 0)
      return "File is empty.";
    return null;
  };

  const handleFile = (f: File) => {
    const err = validateFile(f);
    if (err) { setError(err); return; }
    setFile(f);
    setError(null);
    setResult(null);
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, []);

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(true);
  }, []);

  const onDragLeave = useCallback(() => setDragging(false), []);

  // ── Analyze ──────────────────────────────────────────────────────────────

  const analyze = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      // Call /api/analyze — a same-origin URL. Next.js rewrites it server-side
      // to the backend (configured via BACKEND_URL in next.config.mjs).
      // Same-origin = no CORS, no IPv6/IPv4 conflicts, works in all browsers.
      const res = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `Server error ${res.status}`);
      }
      const data: AnalyzeResponse = await res.json();
      setResult(data);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg + " — make sure the backend is running: uvicorn main:app --host 0.0.0.0 --port 8000");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => { setFile(null); setResult(null); setError(null); };

  // ── Render ───────────────────────────────────────────────────────────────

  const riskColor = (level: "RED" | "YELLOW" | "GREEN") => RISK_COLORS[level];

  const overallColor = result
    ? result.report.overall_score >= 7 ? "#ef5350"
    : result.report.overall_score >= 4 ? "#ffc107"
    : "#66bb6a"
    : "#888";

  return (
    <main style={{ maxWidth: 900, margin: "0 auto", padding: "32px 16px" }}>

      {/* Header */}
      <header style={{ textAlign: "center", marginBottom: 40 }}>
        <h1 style={{ fontSize: 36, fontWeight: 800, color: "#7c8cf8", margin: "0 0 8px" }}>
          ⚖ LEXGUARD
        </h1>
        <p style={{ color: "#9ca3af", fontSize: 16, margin: 0 }}>
          AI-powered contract risk analysis — understand what you are signing
        </p>
      </header>

      {/* Upload zone */}
      {!result && (
        <section aria-label="Document upload">

          {/* The label wraps the hidden input — clicking anywhere inside opens
              the file picker natively without needing JS .click(). This is
              the most browser-compatible approach and works on mobile too. */}
          <label
            htmlFor="contract-file-input"
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            style={{
              display: "block",
              border: `2px dashed ${dragging ? "#7c8cf8" : file ? "#66bb6a" : "#4f46e5"}`,
              borderRadius: 16,
              padding: "40px 24px",
              textAlign: "center",
              cursor: "pointer",
              background: dragging ? "#1a1d2e" : file ? "#0d1f0d" : "#161b27",
              transition: "all 0.2s",
              marginBottom: 16,
            }}
          >
            <input
              id="contract-file-input"
              ref={inputRef}
              type="file"
              accept=".pdf,.docx,.doc"
              style={{ display: "none" }}
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
              aria-label="Select PDF or DOCX file"
            />

            {file ? (
              <>
                <div style={{ fontSize: 40, marginBottom: 8 }}>✅</div>
                <p style={{ color: "#66bb6a", fontWeight: 700, fontSize: 18, margin: "0 0 4px" }}>
                  {file.name}
                </p>
                <p style={{ color: "#9ca3af", fontSize: 13, margin: "0 0 12px" }}>
                  {(file.size / 1024).toFixed(1)} KB · click to change
                </p>
              </>
            ) : (
              <>
                <div style={{ fontSize: 40, marginBottom: 12 }}>📂</div>
                <p style={{ color: "#e8eaf6", fontWeight: 700, fontSize: 18, margin: "0 0 6px" }}>
                  Click here to choose your contract
                </p>
                <p style={{ color: "#6b7280", fontSize: 13, margin: "0 0 16px" }}>
                  or drag and drop · PDF or DOCX · Max 15 MB
                </p>
                <span style={{
                  display: "inline-block",
                  background: "#4f46e5",
                  color: "#fff",
                  padding: "10px 24px",
                  borderRadius: 8,
                  fontWeight: 600,
                  fontSize: 14,
                  pointerEvents: "none",
                }}>
                  Browse files
                </span>
              </>
            )}
          </label>

          {error && (
            <div role="alert" style={{ background: "#3d1a1a", border: "1px solid #ef5350", borderRadius: 8, padding: "12px 16px", color: "#ffcdd2", marginBottom: 16 }}>
              ⚠ {error}
            </div>
          )}

          {/* Analyze button — always visible, state communicates next step */}
          <button
            onClick={file && !loading ? analyze : undefined}
            disabled={!file || loading}
            aria-label={file ? "Analyze contract" : "Select a file first"}
            style={{
              width: "100%",
              padding: "16px",
              borderRadius: 12,
              border: "none",
              background: file && !loading ? "#7c8cf8" : "#1f2937",
              color: file && !loading ? "#fff" : "#4b5563",
              fontSize: 16,
              fontWeight: 700,
              cursor: file && !loading ? "pointer" : "default",
              transition: "all 0.2s",
            }}
          >
            {loading
              ? "⏳ Analyzing with 4 AI agents…"
              : file
              ? "→ Analyze Contract"
              : "↑ Select a file above to begin"}
          </button>

          {loading && (
            <p role="status" aria-live="polite" style={{ textAlign: "center", color: "#9ca3af", marginTop: 16, fontSize: 13 }}>
              Running 4-agent AI pipeline — takes 60–120 seconds for a full contract
            </p>
          )}
        </section>
      )}

      {/* Results */}
      {result && (
        <section aria-label="Analysis results">
          {/* Overall score */}
          <div style={{
            background: "#161b27", border: `2px solid ${overallColor}`,
            borderRadius: 16, padding: "24px", marginBottom: 24, textAlign: "center",
          }}>
            <p style={{ color: "#9ca3af", margin: "0 0 8px", fontSize: 14 }}>
              {result.filename} — {result.report.document_type.replace(/_/g, " ").toUpperCase()}
            </p>
            <div style={{ display: "flex", justifyContent: "center", alignItems: "baseline", gap: 8, marginBottom: 12 }}>
              <span aria-label={`Overall risk score ${result.report.overall_score} out of 10`}
                style={{ fontSize: 56, fontWeight: 900, color: overallColor, lineHeight: 1 }}>
                {result.report.overall_score.toFixed(1)}
              </span>
              <span style={{ color: "#9ca3af", fontSize: 18 }}>/10</span>
              <span style={{
                background: overallColor, color: "#000", fontWeight: 800,
                padding: "4px 12px", borderRadius: 20, fontSize: 14,
              }}>
                {result.report.overall_score >= 7 ? "HIGH RISK" : result.report.overall_score >= 4 ? "MEDIUM RISK" : "LOW RISK"}
              </span>
            </div>

            {/* Count pills */}
            <div style={{ display: "flex", justifyContent: "center", gap: 12, marginBottom: 16 }} role="list" aria-label="Risk level summary">
              {[
                { count: result.report.red_count, label: "HIGH", color: "#ef5350" },
                { count: result.report.yellow_count, label: "MEDIUM", color: "#ffc107" },
                { count: result.report.green_count, label: "LOW", color: "#66bb6a" },
              ].map(({ count, label, color }) => (
                <div key={label} role="listitem" aria-label={`${count} ${label} risk clauses`}
                  style={{ background: "#0f1117", border: `1px solid ${color}`, borderRadius: 8, padding: "8px 16px", textAlign: "center" }}>
                  <div style={{ fontSize: 24, fontWeight: 900, color }}>{count}</div>
                  <div style={{ fontSize: 11, color: "#9ca3af" }}>{label} RISK</div>
                </div>
              ))}
            </div>

            <p style={{ color: "#d1d5db", fontSize: 14, margin: "0 0 8px", lineHeight: 1.6 }}>
              {result.report.executive_summary}
            </p>

            <button onClick={reset} style={{
              marginTop: 8, padding: "8px 20px", borderRadius: 8, border: "1px solid #374151",
              background: "transparent", color: "#9ca3af", cursor: "pointer", fontSize: 14,
            }}>
              ↩ Analyze another contract
            </button>
          </div>

          {/* Clause cards */}
          <h2 style={{ fontSize: 18, fontWeight: 700, color: "#e8eaf6", margin: "0 0 16px" }}>
            Clause-by-clause analysis ({result.report.clauses.length} clauses)
          </h2>

          <div role="list" aria-label="Contract clauses">
            {result.report.clauses.map((clause) => {
              const c = riskColor(clause.risk_level);
              const expanded = expandedId === clause.clause_id;

              return (
                <article key={clause.clause_id} role="listitem"
                  aria-label={`Clause ${clause.clause_id}: ${clause.risk_label} risk, score ${clause.severity_score}`}
                  style={{
                    background: c.bg, border: `1px solid ${c.border}`,
                    borderRadius: 12, marginBottom: 12, overflow: "hidden",
                  }}>

                  {/* Card header — always visible */}
                  <button
                    onClick={() => setExpandedId(expanded ? null : clause.clause_id)}
                    aria-expanded={expanded}
                    style={{
                      width: "100%", background: "none", border: "none",
                      padding: "16px 20px", cursor: "pointer", textAlign: "left",
                      display: "flex", alignItems: "center", gap: 12,
                    }}
                  >
                    {/* Risk badge — color + text label + number all together (accessibility) */}
                    <span aria-label={`${clause.risk_label} severity ${clause.severity_score}`}
                      style={{
                        background: c.badge, color: "#000", fontWeight: 800,
                        padding: "4px 10px", borderRadius: 6, fontSize: 12,
                        whiteSpace: "nowrap", flexShrink: 0,
                      }}>
                      {clause.risk_level} · {clause.severity_score.toFixed(1)} · {clause.risk_label}
                    </span>

                    <span style={{
                      background: "#1f2937", color: "#9ca3af",
                      padding: "2px 8px", borderRadius: 4, fontSize: 11,
                      whiteSpace: "nowrap", flexShrink: 0,
                    }}>
                      {clause.clause_type.replace(/_/g, " ")}
                    </span>

                    {clause.is_predatory && (
                      <span aria-label="Predatory clause" style={{
                        background: "#7f1d1d", color: "#fca5a5",
                        padding: "2px 8px", borderRadius: 4, fontSize: 11, flexShrink: 0,
                      }}>
                        ⚠ PREDATORY
                      </span>
                    )}

                    <span style={{ color: c.text, fontSize: 14, flexGrow: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {clause.original_text.slice(0, 100)}{clause.original_text.length > 100 ? "…" : ""}
                    </span>

                    <span style={{ color: "#6b7280", fontSize: 12, flexShrink: 0 }}>
                      {expanded ? "▲" : "▼"}
                    </span>
                  </button>

                  {/* Expanded details */}
                  {expanded && (
                    <div style={{ padding: "0 20px 20px", borderTop: `1px solid ${c.border}` }}>

                      {/* Original text */}
                      <blockquote style={{
                        background: "#0f1117", borderLeft: `3px solid ${c.border}`,
                        margin: "16px 0", padding: "12px 16px",
                        color: "#9ca3af", fontSize: 13, fontStyle: "italic", lineHeight: 1.6,
                      }}>
                        "{clause.original_text}"
                      </blockquote>

                      {clause.is_ambiguous && clause.ambiguity_note && (
                        <div style={{ background: "#1f1a00", border: "1px solid #ffc107", borderRadius: 8, padding: "10px 14px", marginBottom: 12, fontSize: 13, color: "#fff8e1" }}>
                          <strong>⚡ Ambiguity:</strong> {clause.ambiguity_note}
                        </div>
                      )}

                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 12 }}>
                        <InfoBlock label="Plain English" text={clause.plain_language_explanation} color={c.text} />
                        <InfoBlock label="If you sign this…" text={clause.scenario_consequence} color={c.text} />
                      </div>

                      <InfoBlock label="Benchmark comparison" text={clause.benchmark_comparison} color="#d1d5db" />

                      {/* Recommended action */}
                      <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 12, marginBottom: clause.alternative_wording ? 12 : 0 }}>
                        <span style={{ color: "#9ca3af", fontSize: 13 }}>Recommended action:</span>
                        <span style={{
                          background: ACTION_COLORS[clause.recommended_action],
                          color: "#000", fontWeight: 700, padding: "3px 10px",
                          borderRadius: 6, fontSize: 12, textTransform: "uppercase",
                        }} aria-label={`Recommended action: ${clause.recommended_action}`}>
                          {clause.recommended_action}
                        </span>
                      </div>

                      {clause.pushback_rationale && (
                        <p style={{ color: "#d1d5db", fontSize: 13, marginBottom: 8, lineHeight: 1.6 }}>
                          <strong style={{ color: "#9ca3af" }}>Why push back:</strong> {clause.pushback_rationale}
                        </p>
                      )}

                      {clause.alternative_wording && (
                        <div style={{ background: "#0f2414", border: "1px solid #66bb6a", borderRadius: 8, padding: "10px 14px", fontSize: 13, color: "#e8f5e9", lineHeight: 1.6 }}>
                          <strong>✍ Suggested alternative wording:</strong>
                          <br />
                          "{clause.alternative_wording}"
                        </div>
                      )}

                      {clause.negotiation_tips.length > 0 && (
                        <ul style={{ color: "#d1d5db", fontSize: 13, margin: "12px 0 0", paddingLeft: 20 }}
                          aria-label="Negotiation tips">
                          {clause.negotiation_tips.map((tip, i) => (
                            <li key={i} style={{ marginBottom: 4 }}>{tip}</li>
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

      {/* Legal disclaimer — always visible */}
      <footer role="contentinfo" aria-label="Legal disclaimer"
        style={{
          marginTop: 40, padding: "16px 20px",
          background: "#161b27", border: "1px solid #374151", borderRadius: 12,
          textAlign: "center", color: "#6b7280", fontSize: 12, lineHeight: 1.6,
        }}>
        <strong style={{ color: "#9ca3af" }}>⚖ Legal Disclaimer</strong><br />
        LEXGUARD is an AI-powered tool for informational purposes only. It does <strong>not</strong> constitute legal advice and is <strong>not</strong> a substitute for consultation with a qualified legal professional. Always review contracts with a licensed attorney before signing.
      </footer>
    </main>
  );
}

// ── Sub-component ──────────────────────────────────────────────────────────

function InfoBlock({ label, text, color }: { label: string; text: string; color: string }) {
  return (
    <div style={{ background: "#0f1117", borderRadius: 8, padding: "10px 14px" }}>
      <div style={{ color: "#6b7280", fontSize: 11, fontWeight: 700, marginBottom: 4, textTransform: "uppercase", letterSpacing: 0.5 }}>
        {label}
      </div>
      <div style={{ color, fontSize: 13, lineHeight: 1.6 }}>{text}</div>
    </div>
  );
}
