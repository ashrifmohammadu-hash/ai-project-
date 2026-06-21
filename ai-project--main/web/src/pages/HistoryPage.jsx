import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { fetchHistory } from "../api/history";
import PredictionCard from "../components/PredictionCard";
import ErrorBanner from "../components/ErrorBanner";
import { saveLastResult } from "../utils/format";
import "./HistoryPage.css";

export default function HistoryPage() {
  const [history, setHistory] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

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

  const openItem = (item) => {
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
      <h1 className="page-title">Scan history</h1>
      <p className="page-subtitle">Your past leaf diagnoses</p>

      <ErrorBanner message={error} onClose={() => setError("")} />

      {loading && <p className="muted">Loading…</p>}
      {!loading && history.length === 0 && (
        <div className="empty-state card">
          <p>No scans saved yet.</p>
          <a href="/upload" className="btn btn-primary">
            Scan your first leaf
          </a>
        </div>
      )}

      <div className="history-list">
        {history.map((item) => (
          <PredictionCard
            key={item.id || item.created_at + item.disease}
            item={item}
            onClick={() => openItem(item)}
          />
        ))}
      </div>
    </div>
  );
}
