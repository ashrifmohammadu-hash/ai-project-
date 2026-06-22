import React, { useState } from "react";
import "./AuthPages.css";

export default function DonorSearchPage() {
  const [q, setQ] = useState("");
  const [results, setResults] = useState([]);

  const search = (e) => {
    e && e.preventDefault();
    // Placeholder search: in a real app this should call an API
    if (!q.trim()) return setResults([]);
    setResults([
      { name: "Rohana Perera", blood: "O+", location: "Galle" },
      { name: "Nimal Silva", blood: "A+", location: "Colombo" },
    ]);
  };

  return (
    <div className="page auth-page fade-in">
      <div className="auth-container">
        <form className="card auth-form auth-card" onSubmit={search} role="search" aria-label="Donor search">
          <h1>Search Blood Donors</h1>
          <div className="form-group">
            <label>Search</label>
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Name, blood group, or location" />
          </div>
          <div className="auth-actions">
            <button className="btn btn-primary" onClick={search}>Search</button>
          </div>

          <div style={{ marginTop: 16 }}>
            {results.length === 0 ? (
              <div className="auth-footer">No results</div>
            ) : (
              <ul>
                {results.map((r, i) => (
                  <li key={i}>{r.name} — {r.blood} — {r.location}</li>
                ))}
              </ul>
            )}
          </div>
        </form>
      </div>
    </div>
  );
}
