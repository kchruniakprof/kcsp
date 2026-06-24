import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import AdminPage from "./AdminPage";

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
    json: async () => ({ role: "admin", email: "admin@example.com" }),
  } as Response);
});

afterEach(() => {
  vi.restoreAllMocks();
});

test("renders admin panel heading", () => {
  render(<AdminPage />, { wrapper });
  expect(screen.getByText(/admin panel/i)).toBeInTheDocument();
});

test("renders sign out button", () => {
  render(<AdminPage />, { wrapper });
  expect(screen.getByText(/sign out/i)).toBeInTheDocument();
});
