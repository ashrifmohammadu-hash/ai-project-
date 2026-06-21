export function formatDisease(name) {
  if (!name) return "Unknown";
  return name
    .replace(/___/g, " ")
    .replace(/__/g, " ")
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function confidenceLevel(confidence) {
  if (confidence >= 60) return "high";
  if (confidence >= 45) return "medium";
  return "low";
}

export function formatDate(iso) {
  if (!iso) return "—";
  try {
    const d = new Date(iso.replace(" ", "T"));
    return d.toLocaleString();
  } catch {
    return iso;
  }
}

export const RESULT_STORAGE_KEY = "pv_last_prediction";

/** Persist preview as data URL so it survives navigation (blob URLs break). */
export function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

export function saveLastResult(result, imagePreviewUrl) {
  sessionStorage.setItem(
    RESULT_STORAGE_KEY,
    JSON.stringify({ result, imagePreviewUrl, savedAt: Date.now() })
  );
}

export function loadLastResult() {
  try {
    const raw = sessionStorage.getItem(RESULT_STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}
