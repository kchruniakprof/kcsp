import { render, screen } from "@testing-library/react";
import LoginPage from "./LoginPage";

test("shows Sign in with Google button", () => {
  render(<LoginPage />);
  expect(screen.getByText(/sign in with google/i)).toBeInTheDocument();
});
