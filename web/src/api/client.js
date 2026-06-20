/**
 * API base URL:
 * - Production (served by Flask from web/dist): same origin, paths like /predict
 * - Dev (Vite on :5173): proxy /api → Flask :5000
 * - Override: set VITE_API_BASE_URL in .env (e.g. http://192.168.1.5:5000 for phone on LAN)
 */
export function getApiBase() {
  const env = import.meta.env.VITE_API_BASE_URL?.trim();
  if (env) return env.replace(/\/$/, "");
  if (import.meta.env.DEV) return "/api";
  return "";
}

export async function apiRequest(path, options = {}) {
  const { body, headers: extraHeaders = {}, timeout, signal, ...rest } = options;
  const headers = { ...extraHeaders };

  if (body !== undefined && !(body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }

  const url = `${getApiBase()}${path.startsWith("/") ? path : `/${path}`}`;

  const serializedBody =
    body === undefined
      ? undefined
      : body instanceof FormData
        ? body
        : JSON.stringify(body);

  // Support an optional timeout and external AbortSignal
  const controller = signal ? null : new AbortController();
  const finalSignal = signal || (controller && controller.signal);
  let timeoutId;
  if (!signal && timeout && controller) {
    timeoutId = setTimeout(() => controller.abort(), timeout);
  }

  try {
    const res = await fetch(url, {
      ...rest,
      headers,
      body: serializedBody,
      signal: finalSignal,
    });

    const text = await res.text();
    let data = null;
    try {
      data = text ? JSON.parse(text) : null;
    } catch {
      data = { error: text || res.statusText };
    }

    if (!res.ok) {
      const err = new Error(data?.error || data?.message || `Request failed (${res.status})`);
      err.status = res.status;
      err.data = data;
      throw err;
    }

    return data;
  } catch (err) {
    if (err && err.name === "AbortError") {
      const abortErr = new Error("Request aborted");
      abortErr.status = 0;
      throw abortErr;
    }
    throw err;
  } finally {
    if (timeoutId) clearTimeout(timeoutId);
  }
}

export async function checkHealth() {
  return apiRequest("/health");
}
