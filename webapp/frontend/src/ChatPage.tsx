import { useState, useEffect, useRef } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import ReactMarkdown from "react-markdown";
import TraceDrawer from "./TraceDrawer";

// ── API helpers ──────────────────────────────────────────────────────────────

const BASE = import.meta.env.BASE_URL.replace(/\/$/, "");

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (res.status === 401) {
    window.location.href = "/kcsp/login";
    throw new Error("Unauthenticated");
  }
  if (!res.ok) {
    const d = await res.json().catch(() => ({}));
    throw new Error((d as { detail?: string }).detail ?? `HTTP ${res.status}`);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

interface Thread { id: number; title: string; updated_at: string | null }
interface Message {
  id: number;
  role: "user" | "assistant";
  content_markdown: string | null;
  status: string;
  current_stage: string | null;
  cost_eur: number | null;
  abstained: boolean;
  cached?: boolean;
}

const fetchMe = () => api<{ role: string; email: string }>("/me");
const fetchThreads = () => api<Thread[]>("/threads");
const fetchMessages = (id: number) => api<Message[]>(`/threads/${id}/messages`);
const createThread = (title: string) =>
  api<{ id: number; title: string }>("/threads", { method: "POST", body: JSON.stringify({ title }) });
const sendAsk = (threadId: number, query: string) =>
  api<{ message_id: number }>(`/threads/${threadId}/ask`, { method: "POST", body: JSON.stringify({ query }) });

function fmtEur(eur: number): string {
  if (eur === 0) return "€0";
  const dec = eur < 0.0001 ? 7 : eur < 0.001 ? 6 : eur < 0.01 ? 5 : 4;
  return `€${eur.toFixed(dec)}`;
}

// ── Component ────────────────────────────────────────────────────────────────

export default function ChatPage() {
  const qc = useQueryClient();
  const [activeThread, setActiveThread] = useState<number | null>(null);
  const [input, setInput] = useState("");
  const [pendingMsgId, setPendingMsgId] = useState<number | null>(null);
  const [traceMessageId, setTraceMessageId] = useState<number | null>(null);
  const [optimisticMessages, setOptimisticMessages] = useState<Message[]>([]);
  const [sseStage, setSseStage] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const { data: me } = useQuery({ queryKey: ["me"], queryFn: fetchMe });

  const { data: threads = [] } = useQuery({
    queryKey: ["threads"],
    queryFn: fetchThreads,
    refetchInterval: 30_000,
  });

  const { data: messages = [] } = useQuery({
    queryKey: ["messages", activeThread],
    queryFn: () => fetchMessages(activeThread!),
    enabled: activeThread !== null,
  });

  // SSE: open stream for pending message
  useEffect(() => {
    if (pendingMsgId === null) {
      setSseStage(null);
      return;
    }

    const sse = new EventSource(`${BASE}/chat/${pendingMsgId}/stream`);

    sse.addEventListener("stage", (e: MessageEvent) => {
      try {
        const { stage } = JSON.parse(e.data) as { stage: string };
        setSseStage(stage);
        setOptimisticMessages((prev) =>
          prev.map((m) =>
            m.role === "assistant" ? { ...m, current_stage: stage } : m
          )
        );
      } catch { /* ignore */ }
    });

    sse.addEventListener("done", () => {
      sse.close();
      setOptimisticMessages([]);
      setPendingMsgId(null);
      setSseStage(null);
      qc.invalidateQueries({ queryKey: ["messages", activeThread] });
    });

    sse.onerror = () => {
      sse.close();
      setOptimisticMessages([]);
      setTimeout(() => {
        setPendingMsgId(null);
        qc.invalidateQueries({ queryKey: ["messages", activeThread] });
      }, 1_000);
    };

    return () => { sse.close(); };
  }, [pendingMsgId, activeThread, qc]);

  // scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, optimisticMessages]);

  const [newThreadName, setNewThreadName] = useState<string | null>(null);
  const [menuThreadId, setMenuThreadId] = useState<number | null>(null);
  const [renamingThreadId, setRenamingThreadId] = useState<number | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const newThreadInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (menuThreadId === null) return;
    const close = () => setMenuThreadId(null);
    document.addEventListener("click", close);
    return () => document.removeEventListener("click", close);
  }, [menuThreadId]);

  const newThreadMutation = useMutation({
    mutationFn: (title: string) => createThread(title),
    onSuccess: (t) => {
      qc.invalidateQueries({ queryKey: ["threads"] });
      setActiveThread(t.id);
    },
  });

  const renameMutation = useMutation({
    mutationFn: ({ id, title }: { id: number; title: string }) =>
      api(`/threads/${id}`, { method: "PATCH", body: JSON.stringify({ title }) }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["threads"] });
      setRenamingThreadId(null);
    },
  });

  const askMutation = useMutation({
    mutationFn: (query: string) => sendAsk(activeThread!, query),
    onSuccess: ({ message_id }) => {
      setPendingMsgId(message_id);
    },
    onError: () => {
      setOptimisticMessages([]);
    },
  });

  const handleSend = () => {
    const q = input.trim();
    if (!q || !activeThread || pendingMsgId) return;
    setInput("");
    setOptimisticMessages([
      { id: -1, role: "user", content_markdown: q, status: "done", current_stage: null, cost_eur: null, abstained: false },
      { id: -2, role: "assistant", content_markdown: null, status: "pending", current_stage: null, cost_eur: null, abstained: false },
    ]);
    askMutation.mutate(q);
  };

  const deleteThreadMutation = useMutation({
    mutationFn: (id: number) => api<void>(`/threads/${id}`, { method: "DELETE" }),
    onSuccess: (_, id) => {
      if (activeThread === id) setActiveThread(null);
      qc.invalidateQueries({ queryKey: ["threads"] });
    },
  });

  const isWaiting = pendingMsgId !== null;
  const displayMessages = [...messages, ...optimisticMessages];

  return (
    <div style={{ display: "flex", height: "100vh", fontFamily: "inherit" }}>
      {/* Sidebar */}
      <div style={{
        width: 240,
        background: "var(--ergo-primary)",
        color: "#fff",
        display: "flex",
        flexDirection: "column",
        flexShrink: 0,
      }}>
        {/* ERGO Logo header */}
        <div style={{ padding: "1rem", background: "var(--ergo-primary)" }}>
          <div style={{ color: "#fff", fontWeight: 900, fontSize: "1.25rem", letterSpacing: "0.1em" }}>
            ERGO
          </div>
          <div style={{ color: "rgba(255,255,255,0.7)", fontSize: "0.7rem", marginTop: "0.1rem" }}>
            Knowledge Chat
          </div>
        </div>

        {/* New thread button */}
        <div style={{ padding: "0.75rem 1rem" }}>
          <button
            type="button"
            onClick={() => {
              setNewThreadName("");
              setTimeout(() => newThreadInputRef.current?.focus(), 0);
            }}
            disabled={newThreadMutation.isPending || newThreadName !== null}
            style={{
              width: "100%",
              padding: "0.5rem 0.75rem",
              background: "rgba(255,255,255,0.15)",
              color: "#fff",
              border: "1px solid rgba(255,255,255,0.3)",
              borderRadius: "4px",
              fontWeight: 700,
              fontSize: "0.875rem",
              cursor: "pointer",
              fontFamily: "inherit",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(255,255,255,0.25)")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "rgba(255,255,255,0.15)")}
          >
            + New thread
          </button>
        </div>

        <hr style={{ border: "none", borderTop: "1px solid rgba(255,255,255,0.2)", margin: 0 }} />

        {/* Thread list */}
        <div style={{ overflowY: "auto", flex: 1 }}>
          {newThreadName !== null && (
            <div style={{ padding: "0.5rem 1rem" }}>
              <input
                ref={newThreadInputRef}
                value={newThreadName}
                onChange={(e) => setNewThreadName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    newThreadMutation.mutate(newThreadName.trim() || "New thread");
                    setNewThreadName(null);
                  }
                  if (e.key === "Escape") setNewThreadName(null);
                }}
                onBlur={() => {
                  if (newThreadName.trim()) newThreadMutation.mutate(newThreadName.trim());
                  setNewThreadName(null);
                }}
                placeholder="Thread name… (Enter to confirm)"
                style={{
                  width: "100%",
                  padding: "0.4rem 0.5rem",
                  background: "rgba(255,255,255,0.1)",
                  border: "1px solid rgba(255,255,255,0.4)",
                  color: "#fff",
                  borderRadius: "4px",
                  fontSize: "0.875rem",
                  fontFamily: "inherit",
                  outline: "none",
                }}
              />
            </div>
          )}
          {threads.length === 0 && newThreadName === null && (
            <p style={{ padding: "0.75rem 1rem", fontSize: "0.8rem", color: "rgba(255,255,255,0.6)", margin: 0 }}>
              No threads
            </p>
          )}
          {[...threads].sort((a, b) =>
            (b.updated_at ?? "").localeCompare(a.updated_at ?? "")
          ).map((t) => (
            <div key={t.id} style={{ position: "relative" }}>
              {renamingThreadId === t.id ? (
                <div style={{ padding: "0.5rem 1rem" }}>
                  <input
                    autoFocus
                    value={renameValue}
                    onChange={(e) => setRenameValue(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        if (renameValue.trim()) renameMutation.mutate({ id: t.id, title: renameValue.trim() });
                        else setRenamingThreadId(null);
                      }
                      if (e.key === "Escape") setRenamingThreadId(null);
                    }}
                    onBlur={() => {
                      if (renameValue.trim()) renameMutation.mutate({ id: t.id, title: renameValue.trim() });
                      else setRenamingThreadId(null);
                    }}
                    style={{
                      width: "100%",
                      padding: "0.4rem 0.5rem",
                      background: "rgba(255,255,255,0.1)",
                      border: "1px solid rgba(255,255,255,0.4)",
                      color: "#fff",
                      borderRadius: "4px",
                      fontSize: "0.875rem",
                      fontFamily: "inherit",
                      outline: "none",
                    }}
                  />
                </div>
              ) : (
                <div
                  onClick={() => setActiveThread(t.id)}
                  style={{
                    padding: "0.75rem 1rem",
                    cursor: "pointer",
                    background: activeThread === t.id ? "var(--ergo-primary-dark)" : "transparent",
                    borderLeft: activeThread === t.id ? "3px solid #fff" : "3px solid transparent",
                    fontSize: "0.875rem",
                    display: "flex",
                    alignItems: "center",
                    gap: "0.25rem",
                    transition: "background 0.15s",
                  }}
                  onMouseEnter={(e) => { if (activeThread !== t.id) e.currentTarget.style.background = "rgba(255,255,255,0.1)"; }}
                  onMouseLeave={(e) => { if (activeThread !== t.id) e.currentTarget.style.background = "transparent"; }}
                >
                  <span style={{ flex: 1, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", fontWeight: 700 }}>
                    {t.title}
                  </span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setMenuThreadId(menuThreadId === t.id ? null : t.id);
                    }}
                    title="Thread options"
                    style={{
                      flexShrink: 0, background: "none", border: "none", cursor: "pointer",
                      color: "rgba(255,255,255,0.6)", fontSize: "1.1rem", lineHeight: 1, padding: "0 0.25rem",
                    }}
                  >
                    ⋮
                  </button>
                </div>
              )}
              {menuThreadId === t.id && (
                <div
                  onClick={(e) => e.stopPropagation()}
                  style={{
                    position: "absolute", right: "0.25rem", top: "100%", zIndex: 20,
                    background: "#fff", borderRadius: "4px",
                    boxShadow: "0 2px 8px rgba(0,0,0,0.25)", minWidth: "130px", overflow: "hidden",
                  }}
                >
                  <button
                    onClick={() => {
                      setMenuThreadId(null);
                      setRenamingThreadId(t.id);
                      setRenameValue(t.title);
                    }}
                    style={{
                      display: "block", width: "100%", padding: "0.5rem 1rem",
                      background: "none", border: "none", cursor: "pointer",
                      textAlign: "left", fontSize: "0.875rem", color: "#333",
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.background = "#f0f0f0")}
                    onMouseLeave={(e) => (e.currentTarget.style.background = "none")}
                  >
                    Rename
                  </button>
                  <button
                    onClick={() => {
                      setMenuThreadId(null);
                      if (window.confirm(`Delete "${t.title}"?`)) deleteThreadMutation.mutate(t.id);
                    }}
                    style={{
                      display: "block", width: "100%", padding: "0.5rem 1rem",
                      background: "none", border: "none", cursor: "pointer",
                      textAlign: "left", fontSize: "0.875rem", color: "#c0392b",
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.background = "#fff0f0")}
                    onMouseLeave={(e) => (e.currentTarget.style.background = "none")}
                  >
                    Delete
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>

        <hr style={{ border: "none", borderTop: "1px solid rgba(255,255,255,0.2)", margin: 0 }} />

        {/* Bottom buttons */}
        <div style={{ padding: "0.75rem 1rem", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
          {me?.role === "admin" && (
            <button
              onClick={() => { window.location.href = "/kcsp/admin"; }}
              style={{
                width: "100%", padding: "0.45rem 0.75rem",
                background: "rgba(255,255,255,0.15)", color: "#fff", fontWeight: 700,
                fontSize: "0.875rem", border: "1px solid rgba(255,255,255,0.3)", borderRadius: "4px",
                cursor: "pointer", letterSpacing: "0.01em", fontFamily: "inherit",
              }}
              onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(255,255,255,0.25)")}
              onMouseLeave={(e) => (e.currentTarget.style.background = "rgba(255,255,255,0.15)")}
            >
              Admin panel
            </button>
          )}
          <button
            onClick={async () => {
              await fetch(`${BASE}/auth/logout`, { method: "POST" });
              window.location.href = "/kcsp/login";
            }}
            style={{
              width: "100%", padding: "0.45rem 0.75rem",
              background: "rgba(255,255,255,0.15)", color: "#fff", fontWeight: 700,
              fontSize: "0.875rem", border: "1px solid rgba(255,255,255,0.3)", borderRadius: "4px",
              cursor: "pointer", letterSpacing: "0.01em", fontFamily: "inherit",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(255,255,255,0.25)")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "rgba(255,255,255,0.15)")}
          >
            Sign out
          </button>
        </div>
      </div>

      {/* Main area */}
      <div style={{ flex: 1, display: "flex", flexDirection: "row", overflow: "hidden" }}>
        <div style={{ flex: 1, display: "flex", flexDirection: "column", background: "#ffffff" }}>
          {!activeThread ? (
            <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
              <div style={{ textAlign: "center" }}>
                <h2 style={{ fontSize: "1.25rem", fontWeight: 700, color: "#222", margin: "0 0 0.5rem" }}>
                  Insurance chatbot
                </h2>
                <p style={{ color: "#888", margin: 0, fontSize: "0.95rem" }}>
                  Select a thread or create a new one
                </p>
              </div>
            </div>
          ) : (
            <>
              {/* Messages */}
              <div style={{ flex: 1, overflowY: "auto", padding: "1.5rem" }}>
                {displayMessages.map((m) => (
                  <MessageBubble
                    key={m.id}
                    message={m}
                    onShowTrace={m.role === "assistant" && m.status === "done"
                      ? () => setTraceMessageId(traceMessageId === m.id ? null : m.id)
                      : undefined}
                    traceOpen={traceMessageId === m.id}
                  />
                ))}
                {isWaiting && optimisticMessages.length === 0 && (
                  <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginTop: "0.5rem" }}>
                    <div className="spinner" />
                    <span style={{ fontSize: "0.85rem", color: "#888" }}>
                      {sseStage ?? "processing..."}
                    </span>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Input */}
              <div style={{
                padding: "1rem 1.5rem",
                background: "#fff",
                borderTop: "1px solid #e0e0e0",
                display: "flex", gap: "0.75rem", alignItems: "flex-end",
              }}>
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleSend();
                    }
                  }}
                  placeholder="Ask a question… (Enter = send, Shift+Enter = new line)"
                  disabled={isWaiting}
                  rows={2}
                  style={{
                    flex: 1, resize: "none", padding: "0.625rem 0.75rem",
                    border: "1px solid #c8c8c8", borderRadius: "4px",
                    fontSize: "0.9375rem", fontFamily: "inherit",
                    background: isWaiting ? "#f5f5f5" : "#fff",
                  }}
                />
                <button
                  type="button"
                  onClick={handleSend}
                  disabled={isWaiting || !input.trim()}
                  style={{
                    padding: "0.625rem 1.25rem",
                    background: isWaiting || !input.trim() ? "#ccc" : "var(--ergo-primary)",
                    color: "#fff",
                    border: "none",
                    borderRadius: "4px",
                    fontWeight: 700,
                    fontSize: "0.9375rem",
                    cursor: isWaiting || !input.trim() ? "not-allowed" : "pointer",
                    fontFamily: "inherit",
                    display: "flex",
                    alignItems: "center",
                    gap: "0.4rem",
                    flexShrink: 0,
                  }}
                >
                  {isWaiting ? <div className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} /> : "Send"}
                </button>
              </div>
            </>
          )}
        </div>

        {/* Trace drawer */}
        {traceMessageId !== null && (
          <TraceDrawer
            messageId={traceMessageId}
            onClose={() => setTraceMessageId(null)}
          />
        )}
      </div>
    </div>
  );
}

function MessageBubble({ message: m, onShowTrace, traceOpen }: {
  message: Message;
  onShowTrace?: () => void;
  traceOpen?: boolean;
}) {
  const isUser = m.role === "user";
  return (
    <div style={{
      display: "flex",
      justifyContent: isUser ? "flex-end" : "flex-start",
      marginBottom: "1rem",
    }}>
      <div style={{
        maxWidth: "72%",
        background: isUser ? "var(--ergo-primary)" : "#fff",
        color: isUser ? "#fff" : "#000",
        padding: "0.75rem 1rem",
        borderRadius: isUser ? "12px 12px 2px 12px" : "12px 12px 12px 2px",
        boxShadow: "0 1px 4px rgba(0,0,0,0.08)",
        fontSize: "0.9375rem",
        lineHeight: 1.55,
        whiteSpace: "pre-wrap",
        wordBreak: "break-word",
      }}>
        {m.status === "error" ? (
          <span style={{ color: isUser ? "#ffc0c0" : "#c0392b" }}>{m.content_markdown}</span>
        ) : m.status === "pending" && !m.content_markdown && !isUser ? (
          <span style={{ color: "#888", fontStyle: "italic" }}>
            {m.current_stage ?? "thinking..."}
          </span>
        ) : m.abstained ? (
          <em style={{ color: "#888" }}>
            The system could not answer this question confidently. Please consult a specialist.
          </em>
        ) : isUser ? (
          m.content_markdown ?? ""
        ) : (
          <div className="md">
            <ReactMarkdown>{m.content_markdown ?? ""}</ReactMarkdown>
          </div>
        )}
        <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", marginTop: m.cost_eur != null || m.cached ? "0.4rem" : 0 }}>
          {m.cost_eur != null && !isUser && (
            <span style={{ fontSize: "0.75rem", opacity: 0.6 }}>{fmtEur(m.cost_eur)}</span>
          )}
          {m.cached && !isUser && (
            <span style={{
              fontSize: "0.65rem", padding: "0.1rem 0.35rem",
              background: "#e8f4e8", color: "#2d7d32",
              borderRadius: "3px", fontWeight: 600,
            }}>
              cached
            </span>
          )}
        </div>
      </div>
      {onShowTrace && (
        <div style={{ marginTop: "0.25rem", display: "flex", alignItems: "flex-end" }}>
          <button
            onClick={onShowTrace}
            style={{
              background: "none", border: "none", cursor: "pointer",
              fontSize: "0.75rem", color: traceOpen ? "var(--ergo-primary)" : "#888",
              padding: "0 0.25rem",
            }}
          >
            {traceOpen ? "▲ hide details" : "▼ details"}
          </button>
        </div>
      )}
    </div>
  );
}
