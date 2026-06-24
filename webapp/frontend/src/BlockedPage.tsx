export default function BlockedPage() {
  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "#f2f2f2",
      }}
    >
      <div
        style={{
          background: "#fff",
          padding: "2.5rem 3rem",
          borderRadius: "4px",
          boxShadow: "0 4px 24px rgba(0,0,0,0.08)",
          maxWidth: "400px",
          textAlign: "center",
        }}
      >
        <h1 style={{ fontSize: "1.5rem", fontWeight: 700, margin: "0 0 1rem", color: "#222" }}>
          Access blocked
        </h1>
        <p style={{ color: "#666", margin: "0 0 2rem", fontSize: "0.95rem" }}>
          Your account has been blocked. Please contact an administrator.
        </p>
        <button
          type="button"
          onClick={async () => {
            await fetch("/auth/logout", { method: "POST" });
            window.location.href = "/kcsp/login";
          }}
          style={{
            padding: "0.6rem 1.5rem",
            background: "#fff",
            color: "var(--ergo-primary)",
            border: "1px solid var(--ergo-primary)",
            borderRadius: "4px",
            fontWeight: 700,
            fontSize: "0.9rem",
            cursor: "pointer",
            fontFamily: "inherit",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "var(--ergo-primary)";
            e.currentTarget.style.color = "#fff";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "#fff";
            e.currentTarget.style.color = "var(--ergo-primary)";
          }}
        >
          Sign out
        </button>
      </div>
    </div>
  );
}
