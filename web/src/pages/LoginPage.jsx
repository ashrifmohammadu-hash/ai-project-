import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { login as apiLogin } from "../api/auth";
import "./AuthPages.css";

export default function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const submit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const res = await apiLogin(username, password);
      if (res?.token) {
        localStorage.setItem("auth_token", res.token);
        localStorage.setItem("auth_user", res.user || username);
        navigate("/dashboard");
      } else {
        setError(res?.error || "Login failed");
      }
    } catch (err) {
      setError(err.message || "Login failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page auth-page fade-in">
      <div className="auth-container">
        <form className="card auth-form auth-card" onSubmit={submit}>
          <h1>Sign in</h1>
          <div className="form-group">
            <label>Username</label>
            <input value={username} onChange={(e) => setUsername(e.target.value)} />
          </div>
          <div className="form-group">
            <label>Password</label>
            <div className="password-wrapper">
              <input type={showPassword ? "text" : "password"} value={password} onChange={(e) => setPassword(e.target.value)} />
              <button type="button" className="password-toggle" onClick={() => setShowPassword(!showPassword)} aria-label={showPassword ? "Hide password" : "Show password"}>
                {showPassword ? "\u{1F648}" : "\u{1F441}\uFE0F"}
              </button>
            </div>
          </div>
          {error && <div className="auth-error">{error}</div>}
          <div className="auth-actions">
            <button type="submit" className="btn btn-primary" disabled={loading}>{loading?"Signing in…":"Sign in"}</button>
          </div>
          <div className="auth-footer">
            <button type="button" className="btn-auth-switch" onClick={() => navigate('/register')}>Create an account</button>
          </div>
        </form>
      </div>
    </div>
  );
}
