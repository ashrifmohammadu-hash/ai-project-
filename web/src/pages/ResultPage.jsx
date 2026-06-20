import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import ConfidenceRing from "../components/ConfidenceRing";
import ErrorBanner from "../components/ErrorBanner";
import ExplanationBox from "../components/ExplanationBox";
import { loadLastResult, formatDisease, confidenceLevel } from "../utils/format";
import { submitAnswer } from "../api/predict";
import { apiRequest } from "../api/client";
import "./ResultPage.css";

export default function ResultPage() {
  const [data, setData] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [error, setError] = useState("");
  const [answering, setAnswering] = useState(false);
  const [explanation, setExplanation] = useState(null);
  const [explanationLoading, setExplanationLoading] = useState(false);
  const [explanationError, setExplanationError] = useState("");

  useEffect(() => {
    const stored = loadLastResult();
    if (!stored?.result) {
      setError("No diagnosis data. Please scan a leaf first.");
      return;
    }
    setData(stored.result);
    setImageUrl(stored.imagePreviewUrl);

    if (stored.imagePreviewUrl) {
      setExplanationLoading(true);
      apiRequest("/explain", {
        method: "POST",
        body: {
          image_base64: stored.imagePreviewUrl,
        },
      })
        .then((res) => {
          setExplanation(res.explanation || null);
        })
        .catch((err) => {
          setExplanationError(err.message || "Could not load explanation");
        })
        .finally(() => {
          setExplanationLoading(false);
        });
    }
  }, []);

  if (!data && !error) {
    return (
      <div className="page">
        <div className="spinner" />
      </div>
    );
  }

  const result = data;
  const treatment = result?.treatment || {};
  const questions = result?.clarification_questions || [];
  const level = confidenceLevel(result?.confidence || 0);
  const topTwo = (result?.all_predictions || []).slice(0, 3);
  const isNotFound = result?.disease === "not_found" || result?.plant_type === "unknown";

  const handleAnswer = async (questionIndex, answer) => {
    const preds = result.all_predictions || [];
    const d1 = preds[0]?.disease || result.disease;
    const d2 = preds[1]?.disease || d1;
    setAnswering(true);
    setError("");
    try {
      const updated = await submitAnswer(d1, d2, questionIndex, answer);
      setData((prev) => ({
        ...prev,
        disease: updated.selected_disease,
        display_name: formatDisease(updated.selected_disease),
        confidence: updated.confidence,
        treatment: updated.treatment,
        needs_clarification: false,
        clarification_questions: [],
      }));
    } catch (err) {
      setError(err.message);
    } finally {
      setAnswering(false);
    }
  };

  return (
    <div className="page fade-in">
      <h1 className="page-title">AI Diagnostic Report</h1>
      <p className="page-subtitle">Plant disease analysis result</p>

      <ErrorBanner message={error} onClose={() => setError("")} />

      {imageUrl && (
        <div className="result-image card">
          <img src={imageUrl} alt="Scanned leaf" />
        </div>
      )}

      <div className="card result-main">
        {isNotFound ? (
          <div className="not-found-box">
            <h2 className="result-disease not-found-title">Not Found</h2>
            <p className="not-found-message">
              {result?.treatment || "This image does not match any known plant disease in our database. The plant species may not be supported by the current model."}
            </p>
            <p className="not-found-hint">
              Supported crops: Apple, Banana, Blueberry, Cherry, Chili, Coconut, Corn,
              Grape, Mango, Orange, Papaya, Peach, Pepper, Potato, Raspberry, Rice,
              Soybean, Squash, Strawberry, Tea, Tomato.
            </p>
          </div>
        ) : (
          <>
        <div className={`severity-badge severity-${level}`}>
          {level === "high" ? "CONFIDENT" : level === "medium" ? "MODERATE" : "LOW"}
        </div>
        <h2 className="result-disease">
          {result.display_name || formatDisease(result.disease)}
        </h2>
        {result.plant_type && (
          <p className="result-plant">{result.plant_type}</p>
        )}
        <ConfidenceRing confidence={result.confidence || 0} />

        <ExplanationBox
          explanation={explanation}
          loading={explanationLoading}
          error={explanationError}
        />

        {result.unsupported_plant && (
          <p className="warning-box">
            This leaf may be outside our training set. Results may be unreliable.
          </p>
        )}

        {result.needs_clarification && questions.length > 0 && (
          <div className="clarify-box">
            <p>Answer YES or NO to help confirm the disease:</p>
            {questions.map((q, i) => (
              <div key={i} className="clarify-question">
                <p>{q.question}</p>
                <div className="clarify-buttons">
                  <button
                    type="button"
                    className="btn btn-primary"
                    disabled={answering}
                    onClick={() => handleAnswer(i, "yes")}
                  >
                    Yes
                  </button>
                  <button
                    type="button"
                    className="btn btn-secondary"
                    disabled={answering}
                    onClick={() => handleAnswer(i, "no")}
                  >
                    No
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}

        <section className="treatment-section">
          <h3>Disease information</h3>
          {treatment.symptoms && (
            <div className="info-row">
              <strong>Symptoms</strong>
              <p>{treatment.symptoms}</p>
            </div>
          )}
          {treatment.treatment && (
            <div className="info-row">
              <strong>Treatment</strong>
              <p>{treatment.treatment}</p>
            </div>
          )}
          {treatment.prevention && (
            <div className="info-row">
              <strong>Prevention</strong>
              <p>{treatment.prevention}</p>
            </div>
          )}
          {treatment.fertilizer && (
            <div className="info-row">
              <strong>Fertilizer</strong>
              <p>{treatment.fertilizer}</p>
            </div>
          )}
          {treatment.severity && (
            <div className="info-row">
              <strong>Severity</strong>
              <p>{treatment.severity}</p>
            </div>
          )}
        </section>

        {topTwo.length > 0 && (
          <section className="alt-predictions">
            <h3>Other possibilities</h3>
            <ul>
              {topTwo.map((p, i) => (
                <li key={i}>
                  {p.display_name || formatDisease(p.disease)} — {p.confidence}%
                </li>
              ))}
            </ul>
          </section>
        )}
          </>
        )}
      </div>

      <div className="result-actions">
        <Link to="/upload" className="btn btn-primary">
          Scan another leaf
        </Link>
        <Link to="/history" className="btn btn-secondary">
          View history
        </Link>
      </div>
    </div>
  );
}
