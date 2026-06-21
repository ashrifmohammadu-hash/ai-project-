import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
import { NavLink, useLocation, useNavigate } from "react-router-dom";
import { useTheme } from "../context/ThemeContext";
import "./Navbar.css";
import { logout as apiLogout } from "../api/auth";

const NAV_ITEMS = [
  { to: "/dashboard", end: true, label: "Home", icon: "⌂" },
  { to: "/upload", label: "Scan", icon: "📷" },
  { to: "/history", label: "History", icon: "🕐" },
  { to: "/profile", label: "Settings", icon: "⚙" },
];

function ThemeIcon({ theme }) {
  if (theme === "light") {
    return (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <path
          d="M12 3a1 1 0 0 1 1 1v1a1 1 0 1 1-2 0V4a1 1 0 0 1 1-1Zm0 15a1 1 0 0 1 1 1v1a1 1 0 1 1-2 0v-1a1 1 0 0 1 1-1Zm9-4a1 1 0 0 1-1 1h-1a1 1 0 1 1 0-2h1a1 1 0 0 1 1 1ZM5 12a1 1 0 0 1-1 1H3a1 1 0 1 1 0-2h1a1 1 0 0 1 1 1Zm12.95 6.364a1 1 0 0 1-1.414-1.414l.707-.707a1 1 0 1 1 1.414 1.414l-.707.707ZM7.757 7.757a1 1 0 0 1-1.414-1.414l.707-.707a1 1 0 0 1 1.414 1.414l-.707.707Zm10.607-1.414a1 1 0 0 1 1.414 1.414l-.707.707a1 1 0 0 1-1.414-1.414l.707-.707ZM7.05 16.95a1 1 0 0 1 1.414 1.414l-.707.707A1 1 0 0 1 6.343 17.657l.707-.707ZM12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8Z"
          fill="currentColor"
        />
      </svg>
    );
  }
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M12 3.5a1 1 0 0 1 1 1v.5a1 1 0 1 1-2 0V4.5a1 1 0 0 1 1-1Zm0 14a1 1 0 0 1 1 1v.5a1 1 0 1 1-2 0V18.5a1 1 0 0 1 1-1ZM20.5 12a1 1 0 0 1-1 1h-.5a1 1 0 1 1 0-2h.5a1 1 0 0 1 1 1ZM5 12a1 1 0 0 1-1 1H4a1 1 0 1 1 0-2h1a1 1 0 0 1 1 1Zm10.2 7.2a1 1 0 0 1-1.4-1.4l.35-.35a1 1 0 1 1 1.4 1.4l-.35.35ZM8.45 8.45a1 1 0 0 1-1.4-1.4l.35-.35a1 1 0 0 1 1.4 1.4l-.35.35Zm8.3-1.4a1 1 0 0 1 1.4 1.4l-.35.35a1 1 0 0 1-1.4-1.4l.35-.35ZM8.1 16.1a1 1 0 0 1 1.4 1.4l-.35.35a1 1 0 0 1-1.4-1.4l.35-.35ZM12 7a5 5 0 1 0 0 10 5 5 0 0 0 0-10Z"
        fill="currentColor"
      />
    </svg>
  );
}

export default function Navbar() {
  const { theme, toggle } = useTheme();
  const location = useLocation();
  const navigate = useNavigate();
  const navRef = useRef(null);
  const [scrolled, setScrolled] = useState(false);
  const [pill, setPill] = useState({ left: 0, width: 0 });
  const [authUser, setAuthUser] = useState(() => {
    try {
      return localStorage.getItem("auth_user");
    } catch {
      return null;
    }
  });

  const updatePill = () => {
    const nav = navRef.current;
    const active = nav?.querySelector(".nav-link.active");
    if (!nav || !active) return;
    const navRect = nav.getBoundingClientRect();
    const rect = active.getBoundingClientRect();
    setPill({
      left: rect.left - navRect.left + nav.scrollLeft,
      width: rect.width,
    });
  };

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useLayoutEffect(() => {
    updatePill();
    const t = requestAnimationFrame(updatePill);
    return () => cancelAnimationFrame(t);
  }, [location.pathname]);

  useEffect(() => {
    window.addEventListener("resize", updatePill);
    return () => window.removeEventListener("resize", updatePill);
  }, []);

  return (
    <>
      <header className={`navbar ${scrolled ? "navbar--scrolled" : ""}`}>
        <div className="navbar-glass" />
        <div className="navbar-inner">
          <NavLink to="/dashboard" className="navbar-brand">
            <span className="brand-icon">🌿</span>
            <span className="brand-text">Plant Village AI</span>
          </NavLink>

          <nav className="navbar-links" ref={navRef} aria-label="Main">
            <span
              className="nav-indicator"
              style={{
                width: pill.width,
                transform: `translateX(${pill.left}px)`,
                opacity: pill.width > 0 ? 1 : 0,
              }}
              aria-hidden="true"
            />
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}
              >
                <span className="nav-link-icon" aria-hidden="true">
                  {item.icon}
                </span>
                <span className="nav-link-label">{item.label}</span>
              </NavLink>
            ))}
          </nav>

          <div className="navbar-actions">
            <button
              type="button"
              className="navbar-btn navbar-btn--icon"
              onClick={toggle}
              title={theme === "light" ? "Dark mode" : "Light mode"}
              aria-label="Toggle theme"
            >
              <ThemeIcon theme={theme} />
            </button>

            {authUser ? (
              <button
                type="button"
                className="navbar-btn navbar-btn--logout"
                onClick={async () => {
                  try {
                    const token = localStorage.getItem("auth_token");
                    await apiLogout(token);
                  } catch (e) {
                    /* ignore */
                  }
                  localStorage.removeItem("auth_token");
                  localStorage.removeItem("auth_user");
                  setAuthUser(null);
                  navigate("/");
                }}
              >
                Logout ({authUser})
              </button>
            ) : (
              <button type="button" className="navbar-btn" onClick={() => navigate("/login")}>Login</button>
            )}
          </div>
        </div>
      </header>
    </>
  );
}
