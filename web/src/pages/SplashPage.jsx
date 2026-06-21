import { Link } from "react-router-dom";
import "../styles/global.css";
import "./SplashPage.css";
import Navbar from "../components/Navbar";

export default function SplashPage() {
  return (
    <main className="home-page">
      <Navbar />
      <div className="animated-bg" aria-hidden="true">
        <span className="blob b1" />
        <span className="blob b2" />
        <span className="blob b3" />
        <span className="blob b4" />
      </div>

      <header className="home-hero">
        <div className="hero-content">
          <div className="hero-badge">🌿</div>
          <h1 className="hero-title">Plant Village AI</h1>
          <p className="hero-sub">Fast, mobile-friendly crop disease detection for farmers.</p>

          <div className="hero-ctas">
            <Link to="/login" className="btn btn-primary btn-lg">Get started</Link>
          </div>
        </div>
        <div className="hero-art">
          <svg viewBox="0 0 600 400" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <linearGradient id="g1" x1="0" x2="1">
                <stop offset="0%" stopColor="#78c06a" />
                <stop offset="100%" stopColor="#3aa17a" />
              </linearGradient>
            </defs>
            <rect x="0" y="0" width="600" height="400" rx="20" fill="url(#g1)" />
            <g transform="translate(60,40)">
              <circle cx="120" cy="120" r="80" fill="#fff" opacity="0.06" />
              <g fill="#fff" opacity="0.9">
                <path d="M80 200c20-60 80-60 100 0" stroke="#fff" strokeWidth="3" fill="none" />
                <circle cx="120" cy="120" r="40" />
              </g>
            </g>
          </svg>
        </div>
      </header>

      <section className="features">
        <h2>Why use Plant Village AI?</h2>
        <div className="features-grid">
          <div className="feature">
            <h3>Quick diagnosis</h3>
            <p>Upload a leaf photo and get actionable results in seconds.</p>
          </div>
          <div className="feature">
            <h3>Local crops</h3>
            <p>Models trained for Sri Lanka crops and common local diseases.</p>
          </div>
          <div className="feature">
            <h3>Treatment guidance</h3>
            <p>Recommendations and steps to help recover affected plants.</p>
          </div>
        </div>
      </section>

      <footer className="home-footer">
        <small>© Plant Village AI — demo project</small>
      </footer>
    </main>
  );
}
