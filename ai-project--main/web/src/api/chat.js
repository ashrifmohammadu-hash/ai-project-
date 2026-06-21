import { apiRequest } from "./client";

export async function sendChatMessage(message, history = []) {
  return apiRequest("/chat", {
    method: "POST",
    body: { message, history },
  });
}
