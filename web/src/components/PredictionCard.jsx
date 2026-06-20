import { formatDisease, formatDate, confidenceLevel } from "../utils/format";
import "./PredictionCard.css";

export default function PredictionCard({ item, onClick }) {
  const name = item.display_name || formatDisease(item.disease);
  const level = confidenceLevel(item.confidence || 0);

  return (
    <button type="button" className="prediction-card" onClick={onClick}>
      <div className="prediction-card-main">
        <strong>{name}</strong>
        <span className={`badge badge-${level}`}>
          {(item.confidence ?? 0).toFixed(1)}%
        </span>
      </div>
      {item.plant_type && <span className="prediction-plant">{item.plant_type}</span>}
      {item.created_at && (
        <span className="prediction-date">{formatDate(item.created_at)}</span>
      )}
    </button>
  );
}
