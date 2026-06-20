import "./ExplanationBox.css";

export default function ExplanationBox({ explanation, loading, error }) {
  if (loading) {
    return (
      <div className="explanation-box explanation-loading">
        <div className="explanation-spinner" />
        <p>Generating explanation...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="explanation-box explanation-error">
        <p>{error}</p>
      </div>
    );
  }

  if (!explanation) return null;

  return (
    <div className="explanation-box">
      <h3>What could be the cause?</h3>
      <p>{explanation}</p>
    </div>
  );
}
