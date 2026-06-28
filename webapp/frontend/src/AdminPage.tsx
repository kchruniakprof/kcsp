import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";

// ── API ─────────────────────────────────────────────────────────────────────

const BASE = import.meta.env.BASE_URL.replace(/\/$/, ""); // "/kcsp"

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (res.status === 401) { window.location.href = "/kcsp/login"; throw new Error("401"); }
  if (res.status === 403) { window.location.href = "/kcsp/chat"; throw new Error("403"); }
  if (!res.ok) { const d = await res.json().catch(() => ({})); throw new Error((d as { detail?: string }).detail ?? `HTTP ${res.status}`); }
  if (res.status === 204) return undefined as T;
  return res.json();
}

interface User {
  id: number;
  email: string;
  name: string;
  role: string;
  status: string;
  budget_eur: number;
  spent_eur: number;
  last_active_at: string | null;
}

interface Metrics {
  cost_today: number;
  cost_month: number;
  total_questions: number;
  abstain_rate: number;
  active_count: number;
  pending_count: number;
  kill_switch: boolean;
}

// ── Component ────────────────────────────────────────────────────────────────

export default function AdminPage() {
  const qc = useQueryClient();
  const [error, setError] = useState("");
  const [editBudget, setEditBudget] = useState<Record<number, string>>({});

  const { data: me } = useQuery({
    queryKey: ["me"],
    queryFn: () => api<{ role: string; email: string }>("/me"),
  });

  const { data: users, isLoading: usersLoading } = useQuery({
    queryKey: ["admin-users"],
    queryFn: () => api<User[]>("/admin/users"),
    enabled: me?.role === "admin",
  });

  const { data: metrics } = useQuery({
    queryKey: ["admin-metrics"],
    queryFn: () => api<Metrics>("/admin/metrics"),
    enabled: me?.role === "admin",
    refetchInterval: 30_000,
  });

  const statusMutation = useMutation({
    mutationFn: ({ userId, action }: { userId: number; action: string }) =>
      api(`/admin/users/${userId}/status`, { method: "POST", body: JSON.stringify({ action }) }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["admin-users"] }); qc.invalidateQueries({ queryKey: ["admin-metrics"] }); },
    onError: (e: Error) => setError(e.message),
  });

  const budgetMutation = useMutation({
    mutationFn: ({ userId, budget_eur }: { userId: number; budget_eur: number }) =>
      api(`/admin/users/${userId}/budget`, { method: "POST", body: JSON.stringify({ budget_eur }) }),
    onSuccess: (_, vars) => {
      qc.invalidateQueries({ queryKey: ["admin-users"] });
      setEditBudget((p) => { const n = { ...p }; delete n[vars.userId]; return n; });
    },
    onError: (e: Error) => setError(e.message),
  });

  const killSwitchMutation = useMutation({
    mutationFn: (kill_switch: boolean) =>
      api("/admin/settings", { method: "POST", body: JSON.stringify({ kill_switch }) }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["admin-metrics"] }),
    onError: (e: Error) => setError(e.message),
  });

  if (me && me.role !== "admin") {
    window.location.href = "/kcsp/chat";
    return null;
  }

  const pending = users?.filter((u) => u.status === "pending") ?? [];
  const others = users?.filter((u) => u.status !== "pending") ?? [];

  return (
    <div style={{ minHeight: "100vh", background: "#f2f2f2" }}>
      {/* Top bar */}
      <div style={{
        background: "var(--ergo-primary)", color: "#fff", padding: "0.75rem 2rem",
        display: "flex", justifyContent: "space-between", alignItems: "center",
      }}>
        <h1 style={{ color: "#fff", fontSize: "1.25rem", fontWeight: 700, margin: 0 }}>Admin Panel</h1>
        <div style={{ display: "flex", gap: "1rem", alignItems: "center" }}>
          <span style={{ color: "rgba(255,255,255,0.7)", fontSize: "0.8rem" }}>{me?.email}</span>
          <button
            type="button"
            onClick={() => fetch("/kcsp/auth/logout", { method: "POST" }).then(() => { window.location.href = "/kcsp/login"; })}
            style={topBarBtn}
          >
            Sign out
          </button>
          <button
            type="button"
            onClick={() => { window.location.href = "/kcsp/chat"; }}
            style={topBarBtn}
          >
            Chat
          </button>
        </div>
      </div>

      <div style={{ maxWidth: 1100, margin: "0 auto", padding: "2rem 1.5rem" }}>
        {error && (
          <div style={{
            background: "#fde8e8", color: "#c0392b", padding: "0.75rem 1rem",
            borderRadius: 4, marginBottom: "1.5rem", fontSize: "0.9rem",
            display: "flex", justifyContent: "space-between", alignItems: "center",
          }}>
            <span>{error}</span>
            <button
              onClick={() => setError("")}
              style={{ background: "none", border: "none", cursor: "pointer", color: "#c0392b", fontSize: "1rem", padding: 0 }}
            >
              ✕
            </button>
          </div>
        )}

        {/* Metrics */}
        {metrics && (
          <>
            <h2 style={{ fontSize: "1rem", fontWeight: 700, margin: "0 0 0.75rem", color: "#222" }}>Metrics</h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: "0.75rem", marginBottom: "2rem" }}>
              <MetricBox label="Cost today" value={`${(metrics.cost_today * 100).toFixed(2)} ct`} />
              <MetricBox label="Cost this month" value={`${(metrics.cost_month * 100).toFixed(2)} ct`} />
              <MetricBox label="Total questions" value={String(metrics.total_questions)} />
              <MetricBox label="Abstain rate" value={`${(metrics.abstain_rate * 100).toFixed(1)}%`} />
              <MetricBox label="Active" value={String(metrics.active_count)} />
              <MetricBox label="Pending" value={String(metrics.pending_count)} highlight={metrics.pending_count > 0} />
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginBottom: "2rem" }}>
              <span style={{ fontWeight: 700, fontSize: "0.9rem" }}>Kill Switch:</span>
              <button
                type="button"
                onClick={() => killSwitchMutation.mutate(!metrics.kill_switch)}
                style={{
                  minWidth: 120, padding: "0.4rem 0.75rem",
                  background: metrics.kill_switch ? "var(--ergo-primary)" : "#fff",
                  color: metrics.kill_switch ? "#fff" : "var(--ergo-primary)",
                  border: "1px solid var(--ergo-primary)",
                  borderRadius: 4, fontWeight: 700, fontSize: "0.875rem",
                  cursor: "pointer", fontFamily: "inherit",
                }}
              >
                {metrics.kill_switch ? "ON — disable" : "OFF — enable"}
              </button>
              {metrics.kill_switch && (
                <span style={{ color: "#c0392b", fontSize: "0.875rem", fontWeight: 600 }}>
                  Bot is not answering questions!
                </span>
              )}
            </div>
            <hr style={{ border: "none", borderTop: "1px solid #e0e0e0", margin: "0 0 2rem" }} />
          </>
        )}

        {/* Pending users */}
        {pending.length > 0 && (
          <>
            <h2 style={{ fontSize: "1rem", fontWeight: 700, margin: "0 0 0.75rem", color: "#E07000" }}>
              Pending ({pending.length})
            </h2>
            <UserTable
              users={pending}
              editBudget={editBudget}
              setEditBudget={setEditBudget}
              onStatus={(id, action) => statusMutation.mutate({ userId: id, action })}
              onBudget={(id, b) => budgetMutation.mutate({ userId: id, budget_eur: b })}
            />
            <hr style={{ border: "none", borderTop: "1px solid #e0e0e0", margin: "1.5rem 0" }} />
          </>
        )}

        {/* All users */}
        <h2 style={{ fontSize: "1rem", fontWeight: 700, margin: "0 0 0.75rem", color: "#222" }}>All users</h2>
        {usersLoading ? (
          <div className="spinner" />
        ) : (
          <UserTable
            users={others}
            editBudget={editBudget}
            setEditBudget={setEditBudget}
            onStatus={(id, action) => statusMutation.mutate({ userId: id, action })}
            onBudget={(id, b) => budgetMutation.mutate({ userId: id, budget_eur: b })}
          />
        )}
      </div>
    </div>
  );
}

// ── Sub-components ───────────────────────────────────────────────────────────

const topBarBtn: React.CSSProperties = {
  background: "rgba(255,255,255,0.15)", color: "#fff",
  border: "1px solid rgba(255,255,255,0.3)", borderRadius: 4,
  padding: "0.3rem 0.75rem", fontSize: "0.875rem",
  cursor: "pointer", fontFamily: "inherit", fontWeight: 600,
};

function MetricBox({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div style={{
      background: "#fff", borderRadius: 4, padding: "0.75rem",
      borderLeft: highlight ? "3px solid #E07000" : "3px solid #e0e0e0",
      boxShadow: "0 1px 4px rgba(0,0,0,0.06)",
    }}>
      <div style={{ fontSize: "0.72rem", color: "#888", marginBottom: "0.15rem" }}>{label}</div>
      <div style={{ fontWeight: 700, color: highlight ? "#E07000" : "#222", fontSize: "0.95rem" }}>{value}</div>
    </div>
  );
}

interface UserTableProps {
  users: User[];
  editBudget: Record<number, string>;
  setEditBudget: React.Dispatch<React.SetStateAction<Record<number, string>>>;
  onStatus: (id: number, action: string) => void;
  onBudget: (id: number, budget: number) => void;
}

function UserTable({ users, editBudget, setEditBudget, onStatus, onBudget }: UserTableProps) {
  const navigate = useNavigate();
  if (users.length === 0) return <p style={{ color: "#888", fontSize: "0.875rem" }}>None.</p>;

  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", background: "#fff", borderRadius: 4, boxShadow: "0 1px 4px rgba(0,0,0,0.06)" }}>
        <thead>
          <tr style={{ background: "var(--ergo-primary)", color: "#fff" }}>
            {["ID", "Email", "Role", "Status", "Budget (EUR)", "Spent (ct)", "Last activity", "Actions"].map((h) => (
              <th key={h} style={{ padding: "0.6rem 0.75rem", textAlign: "left", fontWeight: 600, fontSize: "0.8rem", whiteSpace: "nowrap" }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {users.map((u, i) => (
            <tr key={u.id} style={{ background: i % 2 === 0 ? "#fafafa" : "#fff", borderBottom: "1px solid #eee" }}>
              <td style={td}>{u.id}</td>
              <td style={{ ...td, fontFamily: "monospace", fontSize: "0.82rem" }}>{u.email}</td>
              <td style={td}>
                <StatusBadge text={u.role} color={u.role === "admin" ? "var(--ergo-primary)" : "#555"} />
              </td>
              <td style={td}>
                <StatusBadge
                  text={u.status}
                  color={u.status === "active" ? "#006600" : u.status === "pending" ? "#E07000" : "#c00"}
                />
              </td>
              <td style={td}>
                {editBudget[u.id] !== undefined ? (
                  <div style={{ display: "flex", gap: "0.25rem" }}>
                    <input
                      type="number" step="0.1" min="0"
                      value={editBudget[u.id]}
                      onChange={(e) => setEditBudget((p) => ({ ...p, [u.id]: e.target.value }))}
                      style={{ width: 70, padding: "0.2rem 0.4rem", border: "1px solid #c8c8c8", borderRadius: 3, fontSize: "0.8rem" }}
                    />
                    <button style={smallBtn("#006600")} onClick={() => onBudget(u.id, parseFloat(editBudget[u.id]) || 0)}>✓</button>
                    <button style={smallBtn("#888")} onClick={() => setEditBudget((p) => { const n = { ...p }; delete n[u.id]; return n; })}>✕</button>
                  </div>
                ) : (
                  <span style={{ cursor: "pointer", textDecoration: "underline dotted" }}
                    onClick={() => setEditBudget((p) => ({ ...p, [u.id]: String(u.budget_eur) }))}>
                    {u.budget_eur.toFixed(2)}
                  </span>
                )}
              </td>
              <td style={td}>{(u.spent_eur * 100).toFixed(3)}</td>
              <td style={{ ...td, fontSize: "0.75rem", color: "#888", whiteSpace: "nowrap" }}>
                {u.last_active_at ? new Date(u.last_active_at).toLocaleString("en-GB") : "—"}
              </td>
              <td style={{ ...td, whiteSpace: "nowrap" }}>
                <div style={{ display: "flex", gap: "0.3rem", flexWrap: "wrap" }}>
                  {u.status === "pending" && (
                    <button style={smallBtn("#006600")} onClick={() => onStatus(u.id, "approve")}>Approve</button>
                  )}
                  {u.status === "active" && u.role !== "admin" && (
                    <button style={smallBtn("#c00")} onClick={() => onStatus(u.id, "block")}>Block</button>
                  )}
                  {u.status === "blocked" && (
                    <button style={smallBtn("#E07000")} onClick={() => onStatus(u.id, "unblock")}>Unblock</button>
                  )}
                  <button style={smallBtn("var(--ergo-primary)")} onClick={() => navigate(`/admin/users/${u.id}`)}>History</button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function StatusBadge({ text, color }: { text: string; color: string }) {
  return (
    <span style={{
      display: "inline-block", padding: "0.1rem 0.5rem",
      background: color + "18", color, borderRadius: 3,
      fontSize: "0.75rem", fontWeight: 600, textTransform: "uppercase",
    }}>
      {text}
    </span>
  );
}

const td: React.CSSProperties = { padding: "0.55rem 0.75rem", fontSize: "0.85rem", verticalAlign: "middle" };

function smallBtn(bg: string): React.CSSProperties {
  return {
    background: bg, color: "#fff", border: "none", borderRadius: 3,
    padding: "0.2rem 0.5rem", fontSize: "0.75rem", cursor: "pointer", fontWeight: 600,
  };
}
