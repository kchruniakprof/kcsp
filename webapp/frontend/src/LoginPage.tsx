const ERROR_MESSAGES: Record<string, string> = {
  oauth_failed: "Google sign-in failed. Please try again.",
  no_email: "Your Google account did not share an email address.",
  blocked: "Your account has been blocked. Contact an administrator.",
};

function errorMessage(): string | null {
  const code = new URLSearchParams(window.location.search).get("error");
  if (!code) return null;
  return ERROR_MESSAGES[code] ?? "Sign-in failed. Please try again.";
}

export default function LoginPage() {
  const error = errorMessage();

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "var(--ergo-primary)",
      }}
    >
      <div
        style={{
          background: "#fff",
          padding: "2.5rem 3rem",
          borderRadius: "4px",
          borderTop: "4px solid var(--ergo-primary-dark)",
          boxShadow: "0 8px 32px rgba(0,0,0,0.25)",
          width: "100%",
          maxWidth: "400px",
        }}
      >
        <h1
          style={{
            color: "var(--ergo-primary)",
            fontSize: "1.75rem",
            fontWeight: 700,
            margin: "0 0 0.5rem",
            letterSpacing: "-0.01em",
          }}
        >
          Insurance chatbot
        </h1>
        <p style={{ color: "#666", margin: "0 0 2rem", fontSize: "0.95rem" }}>
          Sign in with your Google account to continue.
        </p>

        {error && (
          <div
            style={{
              background: "#fde8e8",
              color: "#c0392b",
              padding: "0.75rem 1rem",
              borderRadius: 4,
              marginBottom: "1.5rem",
              fontSize: "0.9rem",
            }}
          >
            {error}
          </div>
        )}

        <button
          type="button"
          onClick={() => {
            window.location.href = (import.meta.env.BASE_URL.replace(/\/$/, "")) + "/auth/google/login";
          }}
          style={{
            width: "100%",
            padding: "0.7rem 1rem",
            background: "var(--ergo-primary)",
            color: "#fff",
            border: "none",
            borderRadius: "4px",
            fontWeight: 700,
            fontSize: "0.95rem",
            cursor: "pointer",
            transition: "background 0.15s ease",
            fontFamily: "inherit",
          }}
          onMouseEnter={(e) => (e.currentTarget.style.background = "var(--ergo-primary-dark)")}
          onMouseLeave={(e) => (e.currentTarget.style.background = "var(--ergo-primary)")}
        >
          Sign in with Google
        </button>
      </div>
    </div>
  );
}
