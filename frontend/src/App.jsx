import { useState } from "react";
import AuthPage from "./pages/AuthPage";
import LandingPage from "./pages/LandingPage";
import Dashboard from "./pages/Dashboard";
import HistoryPage from "./pages/HistoryPage";
import ImageUploader from "./components/ImageUploader";
import Navbar from "./components/Navbar";
import {
  getCurrentUser,
  isAuthenticated,
  logoutUser,
} from "./services/auth";

function App() {
  const [user, setUser] = useState(getCurrentUser());
  const [page, setPage] = useState("dashboard");
  const [authView, setAuthView] = useState("landing");

  const handleAuthenticated = (authenticatedUser) => {
    setUser(authenticatedUser);
    setPage("dashboard");
    setAuthView("landing");
  };

  const handleLogout = () => {
    logoutUser();
    setUser(null);
    setPage("dashboard");
    setAuthView("landing");
  };

  if (!isAuthenticated() || !user) {
    if (authView === "auth") {
      return (
        <AuthPage
          onAuthenticated={handleAuthenticated}
          onBackToLanding={() => setAuthView("landing")}
        />
      );
    }

    return (
      <LandingPage
        onGetStarted={() => setAuthView("auth")}
        onSignIn={() => setAuthView("auth")}
      />
    );
  }

  return (
    <div className="min-h-screen bg-[#F7F2EB] text-[#2F3728]">
      <Navbar
        user={user}
        currentPage={page}
        onDashboard={() => setPage("dashboard")}
        onAnalysis={() => setPage("analysis")}
        onHistory={() => setPage("history")}
        onLogout={handleLogout}
      />

      {page === "dashboard" ? (
        <Dashboard
          onAnalysis={() => setPage("analysis")}
          onHistory={() => setPage("history")}
        />
      ) : page === "history" ? (
        <HistoryPage />
      ) : (
        <ImageUploader />
      )}
    </div>
  );
}

export default App;
