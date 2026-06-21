import "./LoadingOverlay.css";

export default function LoadingOverlay({ message = "Analyzing leaf…" }) {
  return (
    <div className="loading-overlay">
      <div className="loading-box">
        <div className="spinner spinner-lg" />
        <p>{message}</p>
        <small>This may take 15–30 seconds</small>
      </div>
    </div>
  );
}
