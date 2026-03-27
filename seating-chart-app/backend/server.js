// =============================================================
// Seating Chart Backend — Express API
// =============================================================
// This server handles guest-list uploads (Excel), name searches,
// and a simple admin login.  Data is stored in a local JSON file
// so no external database is needed.
// =============================================================

const express = require("express");
const cors = require("cors");
const multer = require("multer");
const XLSX = require("xlsx");
const fs = require("fs");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 5000;

// --------------- configuration ---------------

// Where we persist the guest list between restarts
const DATA_FILE = path.join(__dirname, "guestData.json");

// Simple admin credentials (change these for your event!)
const ADMIN_PASSWORD = "seating2024";
const ADMIN_TOKEN = "admin-token-2024";

// --------------- middleware ---------------

// Allow requests from any origin (the React frontend runs on a different port)
app.use(cors());

// Parse JSON request bodies (for the login endpoint)
app.use(express.json());

// Configure multer to store uploaded files in memory (not on disk)
const upload = multer({ storage: multer.memoryStorage() });

// --------------- in-memory state ---------------

// We keep the guest list in memory for fast searches and write it to
// guestData.json so it survives server restarts.
let guests = [];
let lastUpdated = null;

// On startup, load any previously-saved guest data
function loadGuestData() {
  try {
    if (fs.existsSync(DATA_FILE)) {
      const raw = fs.readFileSync(DATA_FILE, "utf-8");
      const data = JSON.parse(raw);
      guests = data.guests || [];
      lastUpdated = data.lastUpdated || null;
      console.log(`Loaded ${guests.length} guests from ${DATA_FILE}`);
    }
  } catch (err) {
    console.error("Could not load guest data:", err.message);
  }
}
loadGuestData();

// Helper: persist the current guest list to disk
function saveGuestData() {
  const payload = { guests, lastUpdated };
  fs.writeFileSync(DATA_FILE, JSON.stringify(payload, null, 2), "utf-8");
}

// --------------- auth middleware ---------------

// Checks for a valid Bearer token in the Authorization header
function requireAuth(req, res, next) {
  const authHeader = req.headers.authorization || "";
  const token = authHeader.replace("Bearer ", "");
  if (token !== ADMIN_TOKEN) {
    return res.status(401).json({ error: "Unauthorized" });
  }
  next();
}

// =============================================================
// ROUTES
// =============================================================

// ----- 1. Health check -----
// Quick way to verify the server is running and see how many guests are loaded.
app.get("/api/health", (_req, res) => {
  res.json({
    status: "ok",
    guestCount: guests.length,
    lastUpdated,
  });
});

// ----- 2. Admin login -----
// Accepts a password and returns a simple token if correct.
app.post("/api/admin/login", (req, res) => {
  const { password } = req.body;
  if (password === ADMIN_PASSWORD) {
    return res.json({ success: true, token: ADMIN_TOKEN });
  }
  return res.status(401).json({ error: "Invalid password" });
});

// ----- 3. Upload guest list (Excel / CSV) -----
// Parses the uploaded file, extracts Name + Table columns, and saves them.
app.post("/api/upload", upload.single("file"), async (req, res) => {
  try {
    // Make sure a file was actually included in the request
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    // Parse the Excel/CSV file from the in-memory buffer
    const workbook = XLSX.read(req.file.buffer, { type: "buffer" });

    // Use the first sheet in the workbook
    const sheetName = workbook.SheetNames[0];
    const sheet = workbook.Sheets[sheetName];

    // Convert to an array of objects (each row becomes { column: value })
    const rows = XLSX.utils.sheet_to_json(sheet);

    if (rows.length === 0) {
      return res.status(400).json({ error: "The uploaded file contains no data rows" });
    }

    // Find the Name and Table columns (case-insensitive)
    const sampleRow = rows[0];
    const columns = Object.keys(sampleRow);

    const nameCol = columns.find((c) => c.toLowerCase().trim() === "name");
    const tableCol = columns.find((c) => c.toLowerCase().trim() === "table");

    if (!nameCol || !tableCol) {
      return res.status(400).json({
        error:
          'Could not find required columns. Your file must have columns named "Name" and "Table".',
      });
    }

    // Extract and normalize guest data
    const parsed = rows
      .map((row) => ({
        name: String(row[nameCol] || "").trim(),
        table: String(row[tableCol] || "").trim(),
      }))
      .filter((g) => g.name.length > 0); // skip blank rows

    // Save to memory and disk
    guests = parsed;
    lastUpdated = new Date().toISOString();
    saveGuestData();

    res.json({
      success: true,
      message: `Uploaded ${parsed.length} guests`,
      count: parsed.length,
    });
  } catch (err) {
    console.error("Upload error:", err);
    res.status(500).json({ error: "Failed to process file: " + err.message });
  }
});

// ----- 4. Search guests -----
// Performs a case-insensitive partial match on guest names.
app.get("/api/search", (req, res) => {
  const query = (req.query.name || "").trim();

  if (!query) {
    return res.json({ query: "", results: [], count: 0 });
  }

  const lower = query.toLowerCase();
  const results = guests.filter((g) => g.name.toLowerCase().includes(lower));

  res.json({ query, results, count: results.length });
});

// ----- 5. Full guest list (admin only) -----
app.get("/api/guests", requireAuth, (_req, res) => {
  res.json({ guests, count: guests.length, lastUpdated });
});

// --------------- start server ---------------

app.listen(PORT, () => {
  console.log(`Seating chart API running on http://localhost:${PORT}`);
});
