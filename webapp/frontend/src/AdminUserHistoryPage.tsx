import { useState, useRef, useEffect } from "react";
import { Link, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import TraceDrawer from "./TraceDrawer";

interface HistoryItem {
  assistant_message_id: number;
  question_text: string | null;
  answer_text: string | null;
  answer_status: string;
  abstained: boolean;
  cached: boolean;
  cost_eur: number | null;
  created_at: string | null;
  thread_title: string | null;
}

const BASE = import.meta.env.BASE_URL.replace(/\/$/, "");

async function api<T>(path: string): Promise<T> {
  const res = await fetch(BASE + path);
  if (res.status === 401) { window.location.href = "/kcsp/login"; throw new Error("401"); }
  if (res.status === 403) { window.location.href = "/kcsp/chat"; throw new Error("403"); }
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function fmtEur(eur: number): string {
  if (eur === 0) return "€0";
  const dec = eur < 0.0001 ? 7 : eur < 0.001 ? 6 : eur < 0.01 ? 5 : 4;
  return `€${eur.toFixed(dec)}`;
}

const STATUS_COLORS: Record<string, string> = {
  done: "#006600",
  error: "#c00",
  timeout: "#c00",
  pending: "#888",
};

export default function AdminUserHistoryPage() {
  const { id } = useParams<{ id: string }>();
  const [offset, setOffset] = useState(0);
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [hasMore, setHasMore] = useState(true);
  const [traceMessageId, setTraceMessageId] = useState<number | null>(null);
  const [expandedAnswers, setExpandedAnswers] = useState<Set<number>>(new Set());

  const { data: me } = useQuery({
    queryKey: ["me"],
    queryFn: () => api<{ role: string; email: string }>("/me"),
  });

  const { isLoading, isFetching } = useQuery({
    queryKey: ["admin-history", id, offset],
    queryFn: async () => {
      const data = await api<HistoryItem[]>(`/admin/users/${id}/history?offset=${offset}`);
      setItems((prev) => offset === 0 ? data : [...prev, ...data]);
      if (data.length < 100) setHasMore(false);
      return data;
    },
    enabled: !!id && me?.role === "admin",
    staleTime: 30_000,
  });

  if (me && me.role !== "admin") {
    window.location.href = "/kcsp/chat";
    return null;
  }

  return (
    <div style={{ display: "flex", height: "100vh", flexDirection: "column" }}>
      {/* Top bar */}
      <div style={{
        background: "var(--ergo-primary)", color: "#fff", padding: "0.75rem 2rem",
        display: "flex", alignItems: "center", gap: "1rem", flexShrink: 0,
      }}>
        <Link to="/admin" style={{ color: "rgba(255,255,255,0.7)", fontSize: "0.875rem", textDecoration: "none" }}>
          ← Admin
        </Link>
        <span style={{ color: "#fff", fontWeight: 700, fontSize: "0.95rem" }}>User history — #{id}</span>
      </div>

      <div style={{ flex: 1, display: "flex", flexDirection: "row", overflow: "hidden" }}>
        {/* List */}
        <div style={{ flex: 1, overflowY: "auto", padding: "1.5rem" }}>
          {isLoading && <div className="spinner" />}

          {items.map((item) => {
            const isClickable = item.answer_status === "done";
            const isOpen = traceMessageId === item.assistant_message_id;
            return (
              <div
                key={item.assistant_message_id}
                onClick={() => isClickable
                  ? setTraceMessageId(isOpen ? null : item.assistant_message_id)
                  : undefined}
                style={{
                  padding: "0.75rem 1rem",
                  marginBottom: "0.5rem",
                  background: isOpen ? "#fff5f5" : "#fff",
                  border: isOpen ? "1px solid var(--ergo-primary)" : "1px solid #e0e0e0",
                  borderRadius: 4,
                  cursor: isClickable ? "pointer" : "default",
                  boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
                }}
              >
                <div style={{ display: "flex", gap: "0.75rem", alignItems: "flex-start" }}>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{
                      fontWeight: 600, fontSize: "0.9rem",
                      whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
                    }}>
                      {item.question_text ?? "(no question text)"}
                    </div>
                    {item.thread_title && (
                      <div style={{ fontSize: "0.75rem", color: "#888", marginTop: "0.15rem" }}>
                        {item.thread_title}
                      </div>
                    )}
                    {item.answer_text && (
                      <AnswerPreview
                        text={item.answer_text}
                        expanded={expandedAnswers.has(item.assistant_message_id)}
                        onToggle={(e) => {
                          e.stopPropagation();
                          setExpandedAnswers((prev) => {
                            const next = new Set(prev);
                            next.has(item.assistant_message_id)
                              ? next.delete(item.assistant_message_id)
                              : next.add(item.assistant_message_id);
                            return next;
                          });
                        }}
                      />
                    )}
                  </div>
                  <div style={{ display: "flex", gap: "0.3rem", alignItems: "center", flexShrink: 0 }}>
                    {item.abstained && (
                      <Badge text="ABSTAIN" color="#E07000" />
                    )}
                    {item.cached && (
                      <Badge text="cached" color="#2d7d32" />
                    )}
                    {item.answer_status !== "done" && (
                      <Badge text={item.answer_status.toUpperCase()} color={STATUS_COLORS[item.answer_status] ?? "#888"} />
                    )}
                    {item.cost_eur != null && (
                      <span style={{ fontSize: "0.78rem", color: "#888" }}>{fmtEur(item.cost_eur)}</span>
                    )}
                    {item.created_at && (
                      <span style={{ fontSize: "0.75rem", color: "#aaa", whiteSpace: "nowrap" }}>
                        {new Date(item.created_at).toLocaleString("en-GB")}
                      </span>
                    )}
                    {isClickable && (
                      <span style={{ fontSize: "0.75rem", color: isOpen ? "var(--ergo-primary)" : "#888" }}>
                        {isOpen ? "▲" : "▶"}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            );
          })}

          {!isLoading && items.length === 0 && (
            <p style={{ color: "#888", fontSize: "0.875rem" }}>No history yet.</p>
          )}

          {hasMore && items.length > 0 && (
            <div style={{ marginTop: "1rem", textAlign: "center" }}>
              <button
                type="button"
                disabled={isFetching}
                onClick={() => setOffset((o) => o + 100)}
                style={{
                  padding: "0.5rem 1.25rem",
                  background: "#fff",
                  color: "var(--ergo-primary)",
                  border: "1px solid var(--ergo-primary)",
                  borderRadius: 4, fontWeight: 700, fontSize: "0.875rem",
                  cursor: isFetching ? "not-allowed" : "pointer",
                  fontFamily: "inherit",
                  display: "inline-flex", alignItems: "center", gap: "0.5rem",
                }}
              >
                {isFetching && <div className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} />}
                Load more
              </button>
            </div>
          )}
        </div>

        {/* Trace drawer */}
        {traceMessageId !== null && (
          <TraceDrawer
            messageId={traceMessageId}
            basePath={`${BASE}/admin`}
            onClose={() => setTraceMessageId(null)}
          />
        )}
      </div>
    </div>
  );
}

function AnswerPreview({
  text,
  expanded,
  onToggle,
}: {
  text: string;
  expanded: boolean;
  onToggle: (e: React.MouseEvent) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [overflows, setOverflows] = useState(false);

  useEffect(() => {
    if (ref.current) {
      setOverflows(ref.current.scrollHeight > ref.current.clientHeight + 2);
    }
  }, [text]);

  return (
    <div style={{ marginTop: "0.5rem" }}>
      <div
        ref={ref}
        style={{
          fontSize: "0.82rem",
          color: "#444",
          lineHeight: 1.5,
          whiteSpace: "pre-wrap",
          overflow: "hidden",
          display: "-webkit-box",
          WebkitBoxOrient: "vertical",
          WebkitLineClamp: expanded ? "unset" : 3,
          maxHeight: expanded ? "none" : undefined,
        }}
      >
        {text}
      </div>
      {(overflows || expanded) && (
        <button
          onClick={onToggle}
          style={{
            background: "none", border: "none", cursor: "pointer",
            color: "var(--ergo-primary)", fontSize: "0.75rem", padding: "0.2rem 0",
            fontWeight: 600, fontFamily: "inherit",
          }}
        >
          {expanded ? "Show less ▲" : "Show more ▼"}
        </button>
      )}
    </div>
  );
}

function Badge({ text, color }: { text: string; color: string }) {
  return (
    <span style={{
      display: "inline-block", padding: "0.1rem 0.45rem",
      background: color + "18", color,
      borderRadius: 3, fontSize: "0.72rem", fontWeight: 700,
      textTransform: "uppercase",
    }}>
      {text}
    </span>
  );
}
