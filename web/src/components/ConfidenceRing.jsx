import { confidenceLevel } from "../utils/format";
import "./ConfidenceRing.css";

export default function ConfidenceRing({ confidence }) {
  const level = confidenceLevel(confidence);
  const pct = Math.min(100, Math.max(0, confidence));

  return (
    <div className={`confidence-ring level-${level}`}>
      <div className="confidence-value">{pct.toFixed(1)}%</div>
      <div className="confidence-label">confidence</div>
    </div>
  );
}
