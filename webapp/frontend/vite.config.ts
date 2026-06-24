import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "/kcsp/",
  build: {
    outDir: "../backend/static",
    emptyOutDir: true,
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test-setup.tsx"],
    globals: true,
    typecheck: { tsconfig: "./tsconfig.test.json" },
  },
  server: {
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true },
      "/auth": "http://localhost:8000",
      "/me": "http://localhost:8000",
      "/threads": "http://localhost:8000",
      "/chat": {
        target: "http://localhost:8000",
        bypass: (req) => {
          if (req.headers.accept?.includes("text/event-stream")) return undefined;
          return undefined;
        },
      },
      "/admin": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
});
