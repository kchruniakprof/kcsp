import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "./ergo-tokens.css";
import "./markdown.css";
import App from "./App";

const qc = new QueryClient();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={qc}>
      <BrowserRouter basename="/kcsp">
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>
);
