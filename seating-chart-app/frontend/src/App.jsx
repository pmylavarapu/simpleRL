// =============================================================
// Seating Chart App — Main React Component
// =============================================================
// Three tabs:
//   1. "Find My Table"  – public guest search
//   2. "Admin"          – password-protected upload & guest list
//   3. "QR Code"        – printable QR code (admin-only)
// =============================================================

import React, { useState, useRef, useCallback } from "react";
import { QRCodeCanvas } from "qrcode.react";
import "./App.css";

// Read the backend URL from the environment variable set in .env
const API = process.env.REACT_APP_API_URL || "http://localhost:5000";

// ---- Customise these for your event ----
const EVENT_TITLE = "Welcome to Our Wedding!";
const EVENT_SUBTITLE = "Search your name to find your table";

// =============================================================
// App
// =============================================================
export default function App() {
  // Which tab is currently visible
  const [tab, setTab] = useState("search");

  // Admin auth state
  const [adminToken, setAdminToken] = useState(null);

  // Tabs that are visible — QR tab only shows after admin login
  const tabs = [
    { id: "search", label: "Find My Table" },
    { id: "admin", label: "Admin" },
    ...(adminToken ? [{ id: "qr", label: "QR Code" }] : []),
  ];

  return (
    <div className="app-wrapper">
      {/* ---- Header ---- */}
      <header className="app-header">
        <h1>{EVENT_TITLE}</h1>
        <p>{EVENT_SUBTITLE}</p>
      </header>

      {/* ---- Main card ---- */}
      <div className="card">
        {/* Tab bar */}
        <div className="tabs">
          {tabs.map((t) => (
            <button
              key={t.id}
              className={`tab-btn ${tab === t.id ? "active" : ""}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        {tab === "search" && <SearchTab />}
        {tab === "admin" && (
          <AdminTab token={adminToken} onLogin={setAdminToken} />
        )}
        {tab === "qr" && adminToken && <QRTab />}
      </div>
    </div>
  );
}

// =============================================================
// Tab 1 — Find My Table (public search)
// =============================================================
function SearchTab() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState(null); // null = hasn't searched yet
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSearch = async (e) => {
    e.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) return;

    setLoading(true);
    setError("");
    setResults(null);

    try {
      const res = await fetch(
        `${API}/api/search?name=${encodeURIComponent(trimmed)}`
      );
      if (!res.ok) throw new Error("Server error");
      const data = await res.json();
      setResults(data.results);
    } catch {
      setError("Connection error. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <form onSubmit={handleSearch}>
        <input
          className="input-field mb-12"
          type="text"
          placeholder="Enter your first or last name..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          autoFocus
        />
        <button className="btn-primary" type="submit" disabled={loading}>
          {loading ? "Searching..." : "Search"}
        </button>
      </form>

      {/* Loading */}
      {loading && <p className="status-msg mt-16">Searching...</p>}

      {/* Error */}
      {error && <p className="error-msg mt-16">{error}</p>}

      {/* Results */}
      {results && results.length > 0 && (
        <div className="mt-16">
          {results.map((g, i) => (
            <div className="result-card" key={i}>
              <span className="result-name">{g.name}</span>
              <span className="result-table">Table {g.table}</span>
            </div>
          ))}
        </div>
      )}

      {/* No results */}
      {results && results.length === 0 && (
        <p className="no-results mt-16">
          We couldn't find your name. Please check the spelling or ask a
          coordinator for help.
        </p>
      )}
    </>
  );
}

// =============================================================
// Tab 2 — Admin (login + upload + guest list)
// =============================================================
function AdminTab({ token, onLogin }) {
  return token ? (
    <AdminPanel token={token} />
  ) : (
    <LoginForm onLogin={onLogin} />
  );
}

// ---- Login form ----
function LoginForm({ onLogin }) {
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const res = await fetch(`${API}/api/admin/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      });
      const data = await res.json();
      if (res.ok && data.success) {
        onLogin(data.token);
      } else {
        setError("Incorrect password.");
      }
    } catch {
      setError("Connection error. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleLogin}>
      <p className="section-title">Admin Login</p>
      <input
        className="input-field mb-12"
        type="password"
        placeholder="Enter admin password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
      />
      <button className="btn-primary" type="submit" disabled={loading}>
        {loading ? "Logging in..." : "Log In"}
      </button>
      {error && <p className="error-msg mt-12">{error}</p>}
    </form>
  );
}

// ---- Admin panel (after login) ----
function AdminPanel({ token }) {
  const [uploadMsg, setUploadMsg] = useState("");
  const [uploadErr, setUploadErr] = useState("");
  const [uploading, setUploading] = useState(false);
  const [guests, setGuests] = useState([]);
  const fileRef = useRef(null);

  // Fetch the full guest list from the backend
  const loadGuests = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/guests`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        const data = await res.json();
        setGuests(data.guests || []);
      }
    } catch {
      // silently ignore — guest list just won't display
    }
  }, [token]);

  // Upload handler
  const handleUpload = async () => {
    const file = fileRef.current?.files[0];
    if (!file) {
      setUploadErr("Please select a file first.");
      return;
    }

    setUploading(true);
    setUploadMsg("");
    setUploadErr("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API}/api/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();

      if (res.ok && data.success) {
        setUploadMsg(data.message);
        await loadGuests(); // refresh guest list
      } else {
        setUploadErr(data.error || "Upload failed.");
      }
    } catch {
      setUploadErr("Connection error. Please try again.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <>
      {/* Upload section */}
      <p className="section-title">Upload Guest List (Excel)</p>
      <p className="instructions">
        Your file must have two columns: <strong>Name</strong> and{" "}
        <strong>Table</strong>. Extra columns will be ignored.
      </p>
      <div className="file-input-wrapper">
        <input
          type="file"
          ref={fileRef}
          accept=".xlsx,.xls,.csv"
        />
      </div>
      <button
        className="btn-primary"
        onClick={handleUpload}
        disabled={uploading}
      >
        {uploading ? "Uploading..." : "Upload"}
      </button>
      {uploadMsg && <p className="success-msg mt-12">{uploadMsg}</p>}
      {uploadErr && <p className="error-msg mt-12">{uploadErr}</p>}

      {/* Guest list */}
      {guests.length > 0 && (
        <>
          <p className="section-title mt-20">
            Guest List ({guests.length} guests)
          </p>
          <div className="guest-table-wrapper">
            <table className="guest-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Name</th>
                  <th>Table</th>
                </tr>
              </thead>
              <tbody>
                {guests.map((g, i) => (
                  <tr key={i}>
                    <td>{i + 1}</td>
                    <td>{g.name}</td>
                    <td>{g.table}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* Template guide */}
      <p className="section-title mt-20">Excel Template Guide</p>
      <div className="template-box">
        Your spreadsheet should look like this:
        <code>
          | Name &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| Table |
          <br />
          | ------------- | ----- |
          <br />
          | Jane Smith &nbsp;&nbsp;&nbsp;| 1 &nbsp;&nbsp;&nbsp;|
          <br />
          | John Doe &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| 3 &nbsp;&nbsp;&nbsp;|
          <br />| Maria Garcia &nbsp;| 2 &nbsp;&nbsp;&nbsp;|
        </code>
      </div>
    </>
  );
}

// =============================================================
// Tab 3 — QR Code (admin-only)
// =============================================================
function QRTab() {
  const url = window.location.href.split("?")[0]; // clean URL without query params
  const qrRef = useRef(null);

  // Download the QR code as a PNG image
  const handleDownload = () => {
    const canvas = document.querySelector(".qr-section canvas");
    if (!canvas) return;
    const link = document.createElement("a");
    link.download = "seating-chart-qr.png";
    link.href = canvas.toDataURL("image/png");
    link.click();
  };

  return (
    <div className="qr-section">
      <p className="section-title">Event QR Code</p>
      <p className="instructions">
        Print this QR code and place it at your event entrance so guests can
        scan it to find their table.
      </p>
      <QRCodeCanvas value={url} size={256} ref={qrRef} />
      <p className="qr-url">{url}</p>
      <button className="btn-secondary mt-12" onClick={handleDownload}>
        Download QR Code
      </button>
    </div>
  );
}
