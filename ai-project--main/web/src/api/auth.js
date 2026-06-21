import { apiRequest } from "./client";

export async function login(username, password) {
  return apiRequest("/login", {
    method: "POST",
    body: { username, password },
  });
}

export async function logout(token) {
  return apiRequest("/logout", {
    method: "POST",
    body: { token },
  });
}

export async function register(username, password, full_name) {
  return apiRequest("/register", {
    method: "POST",
    body: { username, password, full_name },
  });
}
