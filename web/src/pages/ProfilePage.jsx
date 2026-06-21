import { useState } from "react";
import { useTheme } from "../context/ThemeContext";
import { checkHealth } from "../api/client";
import ErrorBanner from "../components/ErrorBanner";
import "./ProfilePage.css";

export default function ProfilePage() {
  const { theme, toggle } = useTheme();
  const [health, setHealth] = useState(null);
  const [error, setError] = useState("");

  const checkBackend = async () => {
    setError("");
    try {
      const data = await checkHealth();
      setHealth(data);
    } catch (err) {
      setError(err.message || "Backend offline");
      setHealth(null);
    }
  };

  return (
    <div className="page fade-in">
      <h1 className="page-title">Settings</h1>
      <p className="page-subtitle">App preferences & backend status</p>

      <ErrorBanner message={error} onClose={() => setError("")} />

      <div className="card settings-list">
        <div className="setting-row">
          <span>Dark mode</span>
          <button type="button" className="btn btn-secondary" onClick={toggle}>
            {theme === "dark" ? "On" : "Off"}
          </button>
        </div>
        <div className="setting-row">
          <span>Backend status</span>
          <button type="button" className="btn btn-secondary" onClick={checkBackend}>
            Check
          </button>
        </div>
        {health && (
          <div className="health-info">
            <p>
              <strong>Status:</strong> {health.status}
            </p>
            {health.predict_rules && (
              <p>
                <strong>Rules:</strong> {health.predict_rules}
              </p>
            )}
            {health.auth && (
              <p>
                <strong>Auth:</strong> {health.auth}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
