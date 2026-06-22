-- SQL schema for donor table
CREATE TABLE IF NOT EXISTS donors (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  blood_group TEXT NOT NULL,
  phone TEXT,
  email TEXT,
  location TEXT,
  created_at TEXT DEFAULT (datetime('now'))
);
