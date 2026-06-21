import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import ImageDropzone from "../components/ImageDropzone";
import ErrorBanner from "../components/ErrorBanner";
import LoadingOverlay from "../components/LoadingOverlay";
import CameraCapture from "../components/CameraCapture";
import { uploadAndPredict } from "../api/predict";
import { saveLastResult, fileToDataUrl } from "../utils/format";
import "./UploadPage.css";

export default function UploadPage() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [showCamera, setShowCamera] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, []);

  const handleFile = (f) => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setError("");
  };

  const handleClear = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(null);
    setPreviewUrl(null);
    setError("");
  };

  const handleDiagnose = async () => {
    if (!file) {
      setError("Please select an image first");
      return;
    }
    setError("");
    setLoading(true);
    try {
      const result = await uploadAndPredict(file);
      const imageDataUrl = await fileToDataUrl(file);
      saveLastResult(result, imageDataUrl);
      navigate("/result");
    } catch (err) {
      const msg = err.message || err.toString() || "Prediction failed";
      if (msg.includes("fetch") || msg.includes("Failed")) {
        setError("Cannot reach backend. Start plant-disease-backend (port 5000) on your PC.");
      } else {
        setError(msg);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page fade-in">
      {loading && <LoadingOverlay message="Running AI diagnosis..." />}
      <h1 className="page-title">AI Disease Detector</h1>
      <p className="page-subtitle">Upload a leaf photo and get an AI-powered diagnosis</p>

      <ErrorBanner message={error} onClose={() => setError("")} />

      <div className="card">
        <ImageDropzone
          file={file}
          previewUrl={previewUrl}
          onFileSelect={handleFile}
          onClear={handleClear}
        />

        {!file && (
          <div className="camera-upload-option">
            <button
              type="button"
              className="btn-camera-option"
              onClick={() => setShowCamera(true)}
            >
              📷 Take Photo
            </button>
            <span className="camera-option-hint">Use your device camera</span>
          </div>
        )}

        <button
          type="button"
          className="btn-diagnose"
          disabled={!file || loading}
          onClick={handleDiagnose}
        >
          {loading && <span className="btn-spinner" />}
          {loading ? "Analysing image..." : "Get Diagnosis"}
        </button>
      </div>

      {showCamera && (
        <CameraCapture
          onCapture={(capturedFile) => {
            setShowCamera(false);
            handleFile(capturedFile);
          }}
          onClose={() => setShowCamera(false)}
        />
      )}
    </div>
  );
}
