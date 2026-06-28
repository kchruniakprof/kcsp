import React from "react";
import { useQuery } from "@tanstack/react-query";

// ── Interfaces ───────────────────────────────────────────────────────────────

interface SourceChunk {
  chunk_id: string;
  section_id: number;
  heading: string;
  breadcrumb: string;
  markdown: string;
  score: number;
}

interface QueryExpansionDetail {
  intent: string;
  paraphrases: string[];
  domain_terms: string[];
  sparse_hints?: string[];
  section_types: string[];
  confidence: number | null;
  chain_of_thought: string[];
  model?: string | null;
  tokens_prompt?: number | null;
  tokens_completion?: number | null;
  duration_ms?: number | null;
  cost_eur?: number | null;
}

interface GeneratorDetail {
  model?: string | null;
  tokens_prompt?: number | null;
  tokens_completion?: number | null;
  duration_ms?: number | null;
  cost_eur?: number | null;
  confidence: number | null;
  chain_of_thought: string[];
}

interface CriticDetail {
  verdict: string | null;
  confidence: number | null;
  reasoning: string[];
  chain_of_thought: string[];
  retried: boolean | null;
  used_ensemble: boolean | null;
  model?: string | null;
  tokens_prompt?: number | null;
  tokens_completion?: number | null;
  duration_ms?: number | null;
  cost_eur?: number | null;
}

interface RetrievalDetail {
  detected_tarif: string | null;
  chunks: SourceChunk[];
  cited_sources?: string[];
}

interface TraceData {
  total_cost_eur: number | null;
  total_duration_ms: number | null;
  abstained: boolean;
  cited_sources: string[];
  query_expansion: QueryExpansionDetail | null;
  retrieval: RetrievalDetail | null;
  generator: GeneratorDetail | null;
  critic: CriticDetail | null;
}

// ── Fetch ────────────────────────────────────────────────────────────────────

async function fetchTrace(messageId: number, basePath = ""): Promise<TraceData> {
  const res = await fetch(`${basePath}/messages/${messageId}/trace`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function fmtEur(eur: number): string {
  if (eur === 0) return "€0";
  const dec = eur < 0.0001 ? 7 : eur < 0.001 ? 6 : eur < 0.01 ? 5 : 4;
  return `€${eur.toFixed(dec)}`;
}

// ── Props ────────────────────────────────────────────────────────────────────

interface Props {
  messageId: number;
  onClose: () => void;
  basePath?: string;
}

// ── Main component ───────────────────────────────────────────────────────────

export default function TraceDrawer({ messageId, onClose, basePath = "" }: Props) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["trace", basePath, messageId],
    queryFn: () => fetchTrace(messageId, basePath),
    staleTime: Infinity,
  });

  const totalEur = data?.total_cost_eur ?? null;

  return (
    <div style={{
      width: 420, background: "#fff", borderLeft: "1px solid #e0e0e0",
      display: "flex", flexDirection: "column", height: "100%", flexShrink: 0,
    }}>
      {/* Header */}
      <div style={{
        padding: "1rem 1.25rem", display: "flex",
        justifyContent: "space-between", alignItems: "center",
        borderBottom: "1px solid #e0e0e0", background: "#fafafa",
      }}>
        <div>
          <div style={{ fontWeight: 700, fontSize: "0.95rem" }}>Reasoning trace</div>
          {totalEur !== null && (
            <div style={{ fontSize: "0.75rem", color: "#888", marginTop: "0.15rem" }}>
              Cost: {fmtEur(totalEur)}
              {data?.total_duration_ms != null && ` · ${(data.total_duration_ms / 1000).toFixed(1)} s`}
            </div>
          )}
        </div>
        <button
          type="button"
          onClick={onClose}
          style={{
            background: "none", border: "none", cursor: "pointer",
            fontSize: "1.25rem", color: "#888", lineHeight: 1, padding: "0.25rem",
          }}
          title="Close"
        >
          ✕
        </button>
      </div>

      <div style={{ flex: 1, overflowY: "auto", padding: "0.75rem 1rem" }}>
        {isLoading && <div className="spinner" />}
        {error && (
          <div style={{ color: "#c0392b", fontSize: "0.875rem" }}>No trace data.</div>
        )}

        {data && (
          <>
            {/* Top-level stats */}
            <div style={{
              display: "grid", gridTemplateColumns: "1fr 1fr",
              gap: "0.5rem", marginBottom: "1rem",
            }}>
              <StatBox label="Cost" value={totalEur != null ? fmtEur(totalEur) : "—"} />
              <StatBox label="Time" value={data.total_duration_ms != null ? `${(data.total_duration_ms / 1000).toFixed(1)} s` : "—"} />
              {data.abstained && <StatBox label="Abstain" value="YES" highlight />}
            </div>

            {/* Stage 1: Query Expansion */}
            {data.query_expansion && (
              <QueryExpansionBlock detail={data.query_expansion} />
            )}

            {/* Stage 2: Retrieval */}
            {data.retrieval && (
              <RetrievalBlock detail={data.retrieval} citedSources={data.cited_sources ?? []} />
            )}

            {/* Stage 3: Generator */}
            {data.generator && (
              <GeneratorBlock detail={data.generator} />
            )}

            {/* Stage 4: Critic */}
            {data.critic && (
              <CriticBlock detail={data.critic} />
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ── Sub-components ───────────────────────────────────────────────────────────

function StatBox({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div style={{
      background: "#f5f5f5", borderRadius: 4, padding: "0.5rem 0.6rem",
      borderLeft: highlight ? "3px solid var(--ergo-primary)" : "3px solid #e0e0e0",
    }}>
      <div style={{ fontSize: "0.7rem", color: "#888" }}>{label}</div>
      <div style={{ fontSize: "0.85rem", fontWeight: 700, color: highlight ? "var(--ergo-primary)" : "#222" }}>
        {value}
      </div>
    </div>
  );
}

// Stage block header
function StageHeader({ label, meta }: { label: string; meta?: string }) {
  return (
    <div style={{
      display: "flex", justifyContent: "space-between",
      padding: "0.4rem 0.6rem",
      background: "var(--ergo-primary)", color: "#fff",
      borderRadius: "4px 4px 0 0", fontSize: "0.8rem",
    }}>
      <span style={{ fontWeight: 700 }}>{label}</span>
      {meta && <span style={{ opacity: 0.75 }}>{meta}</span>}
    </div>
  );
}

// Stage 1: Query Expansion
function QueryExpansionBlock({ detail }: { detail: QueryExpansionDetail }) {
  const intentColors: Record<string, string> = {
    FACTUAL: "#2980b9",
    COMPARATIVE: "#8e44ad",
    OPERATIONAL: "#16a085",
    OUT_OF_SCOPE: "#c0392b",
  };
  const intentColor = intentColors[detail.intent] ?? "#555";

  return (
    <div style={{ marginBottom: "1rem" }}>
      <StageHeader
        label="Query Expansion"
        meta={detail.confidence != null ? `${(detail.confidence * 100).toFixed(0)}% confidence` : undefined}
      />
      <div style={{ border: "1px solid #e8e8e8", borderTop: "none", borderRadius: "0 0 4px 4px", overflow: "hidden" }}>
        {/* Model / token info */}
        {detail.model && (
          <div style={{ padding: "0.3rem 0.6rem", background: "#f5f5f5", borderBottom: "1px solid #e8e8e8" }}>
            <span style={{ color: "#888", fontSize: "0.72rem" }}>
              {detail.model}
              {detail.tokens_prompt != null && ` · ${detail.tokens_prompt}+${detail.tokens_completion} tok`}
              {detail.duration_ms != null && ` · ${detail.duration_ms} ms`}
              {detail.cost_eur != null && ` · ${fmtEur(detail.cost_eur)}`}
            </span>
          </div>
        )}
        {/* Intent + section types */}
        <div style={{ padding: "0.4rem 0.6rem", background: "#fafafa", display: "flex", flexWrap: "wrap", gap: "0.3rem", alignItems: "center" }}>
          <span style={{
            padding: "0.15rem 0.5rem", borderRadius: 3,
            background: intentColor, color: "#fff",
            fontSize: "0.72rem", fontWeight: 700, letterSpacing: "0.03em",
          }}>
            {detail.intent}
          </span>
          {detail.section_types.map((s, i) => (
            <span key={i} style={{
              padding: "0.15rem 0.45rem", borderRadius: 3,
              background: "#e8f0fe", color: "#1a56db",
              fontSize: "0.68rem", fontWeight: 600,
            }}>
              {s}
            </span>
          ))}
        </div>

        {/* Paraphrases */}
        {detail.paraphrases.length > 0 && (
          <div style={{ padding: "0.35rem 0.6rem", background: "#fff", borderTop: "1px solid #e8e8e8" }}>
            <div style={{ fontSize: "0.7rem", fontWeight: 700, color: "#555", marginBottom: "0.2rem" }}>Paraphrases</div>
            {detail.paraphrases.map((p, i) => (
              <div key={i} style={{ fontSize: "0.74rem", color: "#333", marginBottom: "0.15rem", display: "flex", gap: "0.35rem" }}>
                <span style={{ color: "var(--ergo-primary)", fontWeight: 700, flexShrink: 0 }}>{i + 1}.</span>
                <span>{p}</span>
              </div>
            ))}
          </div>
        )}

        {/* Domain terms */}
        {detail.domain_terms.length > 0 && (
          <div style={{ padding: "0.35rem 0.6rem", background: "#fafafa", borderTop: "1px solid #e8e8e8", display: "flex", flexWrap: "wrap", gap: "0.25rem", alignItems: "center" }}>
            <span style={{ fontSize: "0.7rem", fontWeight: 700, color: "#555", marginRight: "0.2rem" }}>Terms:</span>
            {detail.domain_terms.map((t, i) => (
              <span key={i} style={{
                padding: "0.1rem 0.4rem", borderRadius: 3,
                background: "#f0f4f0", color: "#2d5a27",
                fontSize: "0.7rem", fontWeight: 600,
              }}>
                {t}
              </span>
            ))}
          </div>
        )}

        {/* Sparse hints */}
        {detail.sparse_hints && detail.sparse_hints.length > 0 && (
          <div style={{ padding: "0.35rem 0.6rem", background: "#fff", borderTop: "1px solid #e8e8e8", display: "flex", flexWrap: "wrap", gap: "0.25rem", alignItems: "center" }}>
            <span style={{ fontSize: "0.7rem", fontWeight: 700, color: "#555", marginRight: "0.2rem" }}>Sparse hints:</span>
            {detail.sparse_hints.map((h, i) => (
              <span key={i} style={{
                padding: "0.1rem 0.4rem", borderRadius: 3,
                background: "#fef9e7", color: "#7d6608",
                fontSize: "0.7rem", fontWeight: 600,
              }}>
                {h}
              </span>
            ))}
          </div>
        )}

        {/* Chain of thought */}
        {detail.chain_of_thought.length > 0 && (
          <details style={{ borderTop: "1px solid #e8e8e8" }}>
            <summary style={{
              fontSize: "0.72rem", cursor: "pointer", color: "#999",
              padding: "0.25rem 0.6rem", background: "#f5f5f5", listStyle: "none",
            }}>
              Justification ({detail.chain_of_thought.length})
            </summary>
            <div style={{ padding: "0.4rem 0.6rem", background: "#fafafa" }}>
              {detail.chain_of_thought.map((c, i) => (
                <div key={i} style={{ fontSize: "0.72rem", color: "#666", marginBottom: "0.15rem" }}>
                  {i + 1}. {c}
                </div>
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  );
}

// Stage 2: Retrieval
function RetrievalBlock({ detail, citedSources }: { detail: RetrievalDetail; citedSources: string[] }) {
  const chunks = detail.chunks ?? [];
  const citedSet = new Set(citedSources.map(String));
  const isCited = (c: SourceChunk) =>
    citedSet.has(String(c.section_id)) || citedSet.has(c.chunk_id);

  return (
    <div style={{ marginBottom: "1rem" }}>
      <StageHeader
        label="Retrieval"
        meta={`${chunks.length} chunk${chunks.length !== 1 ? "s" : ""} · ${citedSources.length} cited`}
      />
      <div style={{ border: "1px solid #e8e8e8", borderTop: "none", borderRadius: "0 0 4px 4px", overflow: "hidden" }}>
        {/* Detected tarif */}
        {detail.detected_tarif && (
          <div style={{ padding: "0.35rem 0.6rem", background: "#fafafa", borderBottom: "1px solid #e8e8e8", display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <span style={{ fontSize: "0.7rem", fontWeight: 700, color: "#555" }}>Detected tarif:</span>
            <span style={{
              padding: "0.1rem 0.5rem", borderRadius: 3,
              background: "var(--ergo-primary)", color: "#fff",
              fontSize: "0.72rem", fontWeight: 700,
            }}>
              {detail.detected_tarif}
            </span>
          </div>
        )}

        {/* Chunks */}
        {chunks.length === 0 ? (
          <div style={{ padding: "0.5rem 0.6rem", fontSize: "0.8rem", color: "#888" }}>No chunks retrieved.</div>
        ) : (
          chunks.map((c, i) => {
            const cited = isCited(c);
            return (
              <div key={i} style={{
                borderTop: i > 0 ? "1px solid #e8e8e8" : "none",
                background: cited ? "#f0faf4" : (i % 2 === 0 ? "#fafafa" : "#fff"),
                borderLeft: cited ? "3px solid #27ae60" : "3px solid transparent",
              }}>
                {/* Header row */}
                <div style={{ padding: "0.35rem 0.6rem", display: "flex", alignItems: "flex-start", gap: "0.4rem" }}>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontWeight: 700, fontSize: "0.76rem", color: "var(--ergo-primary)" }}>{c.heading}</div>
                    <div style={{ color: "#666", fontSize: "0.72rem" }}>{c.breadcrumb}</div>
                    <div style={{ color: "#888", fontSize: "0.72rem" }}>score: {c.score?.toFixed(3)}</div>
                  </div>
                  {cited && (
                    <span style={{
                      padding: "0.1rem 0.45rem", borderRadius: 3,
                      background: "#27ae60", color: "#fff",
                      fontSize: "0.65rem", fontWeight: 800,
                      letterSpacing: "0.04em", flexShrink: 0, marginTop: "0.1rem",
                    }}>
                      CITED
                    </span>
                  )}
                </div>
                {/* Chunk content collapsible */}
                {c.markdown && (
                  <details style={{ borderTop: "1px solid #e8e8e8" }}>
                    <summary style={{
                      fontSize: "0.7rem", cursor: "pointer", color: "#999",
                      padding: "0.2rem 0.6rem", background: "#f5f5f5", listStyle: "none",
                    }}>
                      Content
                    </summary>
                    <div style={{
                      padding: "0.4rem 0.6rem", background: "#fafafa",
                      fontSize: "0.72rem", color: "#333",
                      whiteSpace: "pre-wrap", fontFamily: "monospace",
                      maxHeight: 240, overflowY: "auto",
                    }}>
                      {c.markdown}
                    </div>
                  </details>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

// Stage 3: Generator
function GeneratorBlock({ detail }: { detail: GeneratorDetail }) {
  const cot = detail.chain_of_thought ?? [];
  return (
    <div style={{ marginBottom: "1rem" }}>
      <StageHeader
        label="Generator"
        meta={detail.duration_ms ? `${detail.duration_ms} ms` : undefined}
      />
      <div style={{ border: "1px solid #e8e8e8", borderTop: "none", borderRadius: "0 0 4px 4px", overflow: "hidden" }}>
        <div style={{ padding: "0.45rem 0.75rem", background: "#fafafa" }}>
          <div style={{ color: "#888", fontSize: "0.75rem" }}>
            {detail.model}
            {detail.tokens_prompt != null && ` · ${detail.tokens_prompt}+${detail.tokens_completion} tok`}
            {detail.duration_ms != null && ` · ${detail.duration_ms} ms`}
            {detail.cost_eur != null && ` · ${fmtEur(detail.cost_eur)}`}
            {detail.confidence != null && ` · ${(detail.confidence * 100).toFixed(0)}% confidence`}
          </div>
        </div>

        {cot.length > 0 && (
          <details style={{ borderTop: "1px solid #e8e8e8" }}>
            <summary style={{
              fontSize: "0.72rem", cursor: "pointer", color: "#999",
              padding: "0.25rem 0.6rem", background: "#f5f5f5", listStyle: "none",
            }}>
              Chain of thought ({cot.length})
            </summary>
            <div style={{ padding: "0.4rem 0.6rem", background: "#f0faf4" }}>
              {cot.map((c, i) => (
                <div key={i} style={{ fontSize: "0.74rem", color: "#1a4a2a", marginBottom: "0.15rem" }}>
                  {i + 1}. {c}
                </div>
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  );
}

// Stage 4: Critic
function CriticBlock({ detail }: { detail: CriticDetail }) {
  const verdict = detail.verdict ?? "—";
  const isAbstain = verdict === "ABSTAIN";
  const accentColor = isAbstain ? "#c0392b" : "#27ae60";
  const reasoning = detail.reasoning ?? [];
  const cot = detail.chain_of_thought ?? [];
  const confidence = detail.confidence;
  const flags: string[] = [];
  if (detail.retried) flags.push("retried");
  if (detail.used_ensemble) flags.push("ensemble");

  const summaryMeta = [
    confidence != null && `${(confidence * 100).toFixed(0)}% confidence`,
    ...flags,
  ].filter(Boolean).join(" · ");

  return (
    <div style={{ marginBottom: "1rem" }}>
      {/* Stage header — use ERGO primary but add verdict badge */}
      <div style={{
        display: "flex", justifyContent: "space-between",
        padding: "0.4rem 0.6rem",
        background: "var(--ergo-primary)", color: "#fff",
        borderRadius: "4px 4px 0 0", fontSize: "0.8rem",
      }}>
        <span style={{ fontWeight: 700 }}>Critic</span>
        <span style={{ opacity: 0.75 }}>{summaryMeta || undefined}</span>
      </div>

      <div style={{ border: "1px solid #e8e8e8", borderTop: "none", borderRadius: "0 0 4px 4px", overflow: "hidden" }}>
        {/* Verdict badge + model info */}
        <div style={{ padding: "0.4rem 0.6rem", background: "#fafafa", display: "flex", alignItems: "center", gap: "0.5rem", flexWrap: "wrap" }}>
          <span style={{
            padding: "0.2rem 0.6rem", borderRadius: 3,
            background: accentColor, color: "#fff",
            fontSize: "0.75rem", fontWeight: 700, letterSpacing: "0.03em",
          }}>
            {verdict}
          </span>
          {flags.map((f, i) => (
            <span key={i} style={{
              padding: "0.1rem 0.4rem", borderRadius: 3,
              background: "#f0f0f0", color: "#555",
              fontSize: "0.68rem", fontWeight: 600,
            }}>
              {f}
            </span>
          ))}
          {detail.model && (
            <span style={{ color: "#888", fontSize: "0.72rem", marginLeft: "auto" }}>
              {detail.model}
              {detail.tokens_prompt != null && ` · ${detail.tokens_prompt}+${detail.tokens_completion} tok`}
              {detail.duration_ms != null && ` · ${detail.duration_ms} ms`}
              {detail.cost_eur != null && ` · ${fmtEur(detail.cost_eur)}`}
            </span>
          )}
        </div>

        {/* Reasoning bullets */}
        {reasoning.length > 0 && (
          <div style={{ padding: "0.4rem 0.6rem", background: "#fff", borderTop: "1px solid #e8e8e8" }}>
            {reasoning.map((r, i) => (
              <div key={i} style={{
                display: "flex", gap: "0.4rem", marginBottom: "0.2rem",
                fontSize: "0.78rem", color: "#333",
              }}>
                <span style={{ color: accentColor, fontWeight: 700, flexShrink: 0 }}>•</span>
                <span>{r}</span>
              </div>
            ))}
          </div>
        )}

        {/* Chain of thought */}
        {cot.length > 0 && (
          <details style={{ borderTop: "1px solid #e8e8e8" }}>
            <summary style={{
              fontSize: "0.72rem", cursor: "pointer", color: "#999",
              padding: "0.25rem 0.6rem", background: "#f5f5f5", listStyle: "none",
            }}>
              Justification ({cot.length})
            </summary>
            <div style={{ padding: "0.4rem 0.6rem", background: "#fafafa" }}>
              {cot.map((c, i) => (
                <div key={i} style={{ fontSize: "0.72rem", color: "#666", marginBottom: "0.15rem" }}>
                  {i + 1}. {c}
                </div>
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  );
}
