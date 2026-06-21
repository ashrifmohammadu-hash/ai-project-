import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { register as apiRegister } from "../api/auth";
import "./AuthPages.css";

export default function RegisterPage() {
  const [username, setUsername] = useState("");
  const [fullName, setFullName] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [usernameTouched, setUsernameTouched] = useState(false);
  const [passwordTouched, setPasswordTouched] = useState(false);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  const usernameValid = emailRegex.test(username);
  const passwordValid = password.length >= 8;
  const fullNameValid = fullName.trim().length > 0;
  const formValid = usernameValid && passwordValid && fullNameValid;

  const submit = async (e) => {
    e.preventDefault();
    setError("");
    setUsernameTouched(true);
    setPasswordTouched(true);
    if (!formValid) {
      setError("Please enter a valid email, full name and a password (min 8 characters)");
      return;
    }
    setLoading(true);
    try {
      const res = await apiRegister(username, password, fullName);
      if (res?.token) {
        localStorage.setItem("auth_token", res.token);
        localStorage.setItem("auth_user", res.user || username);
        navigate("/dashboard");
      } else {
        setError(res?.error || "Registration failed");
      }
    } catch (err) {
      setError(err.message || "Registration failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page auth-page fade-in">
      <div className="auth-container">
        <form className="card auth-form auth-card" onSubmit={submit}>
          <h1>Create account</h1>

          <div className="form-group">
            <label>Username</label>
            <input
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              onBlur={() => setUsernameTouched(true)}
              placeholder="you@example.com"
              className={usernameTouched ? (usernameValid ? 'input-valid' : 'input-invalid') : ''}
            />
            {usernameTouched && !usernameValid && (
              <div className="auth-error">Enter a valid email address</div>
            )}
          </div>

          <div className="form-group">
            <label>Full name</label>
            <input
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              placeholder="Your full name"
            />
            {!fullNameValid && <div className="auth-error">Enter your full name</div>}
          </div>

          <div className="form-group">
            <label>Password</label>
            <div className="password-wrapper">
              <input
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                onBlur={() => setPasswordTouched(true)}
                className={passwordTouched ? (passwordValid ? 'input-valid' : 'input-invalid') : ''}
              />
              <button type="button" className="password-toggle" onClick={() => setShowPassword(!showPassword)} aria-label={showPassword ? "Hide password" : "Show password"}>
                {showPassword ? "\u{1F648}" : "\u{1F441}\uFE0F"}
              </button>
            </div>
            {passwordTouched && !passwordValid && (
              <div className="auth-error">Password must be at least 8 characters</div>
            )}
          </div>

          {error && <div className="auth-error">{error}</div>}
          <div className="auth-actions">
            <button
              type="submit"
              className="btn-create"
              disabled={loading || !formValid}
              aria-busy={loading}
            >
              {loading ? "Creating…" : "Create account"}
            </button>
          </div>
          <div className="username-hint">Use your email as username (e.g. you@example.com)</div>
          <div className="auth-footer">
            <button type="button" className="btn-auth-switch" onClick={() => navigate('/login')}>Already have an account? Sign in</button>
          </div>
        </form>
      </div>
    </div>
  );
}
