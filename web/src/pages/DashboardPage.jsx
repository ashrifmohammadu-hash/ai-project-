import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { fetchHistory } from "../api/history";
import PredictionCard from "../components/PredictionCard";
import ErrorBanner from "../components/ErrorBanner";
import { saveLastResult } from "../utils/format";
import "./DashboardPage.css";

export default function DashboardPage() {
  const navigate = useNavigate();
  const [history, setHistory] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await fetchHistory();
        if (!cancelled) setHistory(data);
      } catch (err) {
        if (!cancelled) setError(err.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const total = history.length;
  const healthy = history.filter((h) =>
    (h.disease || "").toLowerCase().includes("healthy")
  ).length;
  const diseased = total - healthy;

  const openHistoryItem = (item) => {
    saveLastResult(
      {
        disease: item.disease,
        display_name: item.display_name,
        plant_type: item.plant_type,
        confidence: item.confidence,
        treatment: item.treatment,
        all_predictions: [],
        needs_clarification: false,
      },
      null
    );
    navigate("/result");
  };

  return (
    <div className="page fade-in">
      <h1 className="page-title">Hello, Farmer 👋</h1>
      <p className="page-subtitle">
        AI disease detection for Sri Lanka crops — banana, rice, coconut, tea, mango & more
      </p>

      <ErrorBanner message={error} onClose={() => setError("")} />

      <div className="grid-stats">
        <div className="stat-card">
          <div className="value">{total}</div>
          <div className="label">Total scans</div>
        </div>
        <div className="stat-card">
          <div className="value">{healthy}</div>
          <div className="label">Healthy</div>
        </div>
        <div className="stat-card">
          <div className="value">{diseased}</div>
          <div className="label">Diseased</div>
        </div>
      </div>

      <Link to="/upload" className="scan-cta btn btn-primary">
        📷 Scan a leaf now
      </Link>

      <section className="recent-section">
        <h2>Recent diagnoses</h2>
        {loading && <p className="muted">Loading history…</p>}
        {!loading && history.length === 0 && (
          <p className="muted">No scans yet. Upload a leaf photo to get started.</p>
        )}
        <div className="recent-list">
          {history.slice(0, 5).map((item) => (
            <PredictionCard
              key={item.id || item.created_at + item.disease}
              item={item}
              onClick={() => openHistoryItem(item)}
            />
          ))}
        </div>
      </section>
    </div>
  );
}
