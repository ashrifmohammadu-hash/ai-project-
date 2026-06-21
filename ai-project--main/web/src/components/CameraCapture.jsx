import { useRef, useState, useEffect, useCallback } from "react";
import "./CameraCapture.css";

export default function CameraCapture({ onCapture, onClose }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [cameraError, setCameraError] = useState("");
  const [captured, setCaptured] = useState(false);
  const [capturedUrl, setCapturedUrl] = useState(null);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function start() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        if (!cancelled) {
          setCameraError(
            err.name === "NotAllowedError"
              ? "Camera access denied. Please allow camera permissions or use file upload."
              : err.name === "NotFoundError"
              ? "No camera found on this device. Please use file upload instead."
              : `Camera unavailable: ${err.message}. Please use file upload.`
          );
        }
      }
    }
    start();
    return () => {
      cancelled = true;
      stopCamera();
    };
  }, [stopCamera]);

  const handleCapture = () => {
    const video = videoRef.current;
    if (!video) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
    setCaptured(true);
    setCapturedUrl(dataUrl);
  };

  const handleConfirm = () => {
    if (!capturedUrl) return;
    stopCamera();
    const blob = dataURLToBlob(capturedUrl);
    const file = new File([blob], "camera-capture.jpg", { type: "image/jpeg" });
    onCapture(file);
  };

  const handleRetake = () => {
    setCaptured(false);
    setCapturedUrl(null);
  };

  return (
    <div className="camera-overlay">
      <div className="camera-panel">
        <button className="camera-close" onClick={() => { stopCamera(); onClose(); }} aria-label="Close camera">×</button>

        {cameraError ? (
          <div className="camera-error">
            <p>{cameraError}</p>
            <button className="btn-camera-fallback" onClick={() => { stopCamera(); onClose(); }}>
              Back to file upload
            </button>
          </div>
        ) : captured ? (
          <div className="camera-captured-view">
            <img src={capturedUrl} alt="Captured" className="camera-preview-img" />
            <div className="camera-actions">
              <button className="btn-camera btn-camera-secondary" onClick={handleRetake}>Retake</button>
              <button className="btn-camera btn-camera-primary" onClick={handleConfirm}>Use Photo</button>
            </div>
          </div>
        ) : (
          <div className="camera-live-view">
            <video ref={videoRef} autoPlay playsInline muted className="camera-video" />
            <button className="btn-camera btn-camera-capture" onClick={handleCapture}>
              📷 Capture
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function dataURLToBlob(dataUrl) {
  const [meta, base64] = dataUrl.split(",");
  const mime = meta.match(/:(.*?);/)[1];
  const bytes = atob(base64);
  const arr = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) {
    arr[i] = bytes.charCodeAt(i);
  }
  return new Blob([arr], { type: mime });
}
