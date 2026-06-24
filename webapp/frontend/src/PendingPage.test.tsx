import { render, screen } from "@testing-library/react";
import PendingPage from "./PendingPage";

test("shows awaiting approval message", () => {
  render(<PendingPage />);
  expect(screen.getByText(/awaiting approval/i)).toBeInTheDocument();
});

test("shows sign out button", () => {
  render(<PendingPage />);
  expect(screen.getByText(/sign out/i)).toBeInTheDocument();
});
