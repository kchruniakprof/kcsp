import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import TraceDrawer from "./TraceDrawer";

function wrapper({ children }: { children: React.ReactNode }) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
}

beforeEach(() => {
  global.fetch = vi.fn().mockResolvedValue({
    ok: false,
    status: 404,
    json: async () => ({}),
  } as Response);
});

afterEach(() => {
  vi.restoreAllMocks();
});

test("renders trace drawer header", () => {
  render(<TraceDrawer messageId={1} onClose={vi.fn()} />, { wrapper });
  expect(screen.getByText(/reasoning trace/i)).toBeInTheDocument();
});

test("renders close button", () => {
  render(<TraceDrawer messageId={1} onClose={vi.fn()} />, { wrapper });
  expect(screen.getByTitle(/close/i)).toBeInTheDocument();
});
