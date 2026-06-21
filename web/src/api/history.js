import { apiRequest } from "./client";

export async function fetchHistory() {
  const data = await apiRequest("/history");
  return Array.isArray(data) ? data : data?.history || data?.items || [];
}
