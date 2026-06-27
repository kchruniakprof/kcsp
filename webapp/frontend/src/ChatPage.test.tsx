import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import ChatPage from "./ChatPage";

function wrapper({ children }: { children: React.ReactNode }) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return (
    <QueryClientProvider client={qc}>
      <MemoryRouter>{children}</MemoryRouter>
    </QueryClientProvider>
  );
}

beforeEach(() => {
  global.fetch = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => ({ role: "user", email: "test@example.com" }),
  } as Response);
});

afterEach(() => {
  vi.restoreAllMocks();
});

test("shows no-thread placeholder", () => {
  render(<ChatPage />, { wrapper });
  expect(screen.getByText(/select a thread or create a new one/i)).toBeInTheDocument();
});

test("shows new thread button", () => {
  render(<ChatPage />, { wrapper });
  expect(screen.getByText(/\+ new thread/i)).toBeInTheDocument();
});

test("shows ERGO branding", () => {
  render(<ChatPage />, { wrapper });
  expect(screen.getByText("ERGO")).toBeInTheDocument();
});
