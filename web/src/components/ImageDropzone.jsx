import { useRef, useState } from "react";
import "./ImageDropzone.css";

export default function ImageDropzone({ file, previewUrl, onFileSelect, onClear }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFiles = (files) => {
    const f = files?.[0];
    if (!f || !f.type.startsWith("image/")) return;
    onFileSelect(f);
  };

  return (
    <div className="dropzone-wrap">
      {previewUrl ? (
        <div className="dropzone-preview">
          <img src={previewUrl} alt="Leaf preview" />
          <button type="button" className="dropzone-clear" onClick={onClear}>
            ×
          </button>
        </div>
      ) : (
        <div
          className={`dropzone ${dragOver ? "dropzone-active" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            handleFiles(e.dataTransfer.files);
          }}
          onClick={() => inputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
        >
          <span className="dropzone-icon">📷</span>
          <p>Drag & drop a leaf photo here</p>
          <span className="dropzone-hint">or click to browse (JPG, PNG)</span>
        </div>
      )}
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        hidden
        onChange={(e) => handleFiles(e.target.files)}
      />
      {file && !previewUrl && <p className="file-name">{file.name}</p>}
    </div>
  );
}
