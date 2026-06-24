import { Routes, Route, Navigate } from "react-router-dom";
import LoginPage from "./LoginPage";
import PendingPage from "./PendingPage";
import BlockedPage from "./BlockedPage";
import ChatPage from "./ChatPage";
import AdminPage from "./AdminPage";
import AdminUserHistoryPage from "./AdminUserHistoryPage";

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/pending" element={<PendingPage />} />
      <Route path="/blocked" element={<BlockedPage />} />
      <Route path="/chat" element={<ChatPage />} />
      <Route path="/admin" element={<AdminPage />} />
      <Route path="/admin/users/:id" element={<AdminUserHistoryPage />} />
      <Route path="/" element={<Navigate to="/login" replace />} />
    </Routes>
  );
}
