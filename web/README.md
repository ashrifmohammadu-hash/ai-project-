# Plant Disease Detector — Web UI

React frontend for **plant-disease-backend** (Flask). No separate web server — the Flask app serves both the API and this UI.

## Quick start (one server)

```powershell
cd "D:\ai data\Final\web"
npm install
npm run build

cd "D:\ai data\Final\plant-disease-backend"
..\Merge-Project\plant_env\Scripts\python.exe app.py
```

Open **http://localhost:5000** — scan leaves, view history, settings.

## Development (hot reload)

Terminal 1 — Flask API:

```powershell
cd "D:\ai data\Final\plant-disease-backend"
..\Merge-Project\plant_env\Scripts\python.exe app.py
```

Terminal 2 — Vite dev server:

```powershell
cd "D:\ai data\Final\web"
npm run dev
```

Open **http://localhost:5173** (API calls go through `/api` → Flask on port 5000).

## Environment

| Variable | When to set |
|----------|-------------|
| `VITE_API_BASE_URL` | Usually **empty**. Set only for LAN phone testing, e.g. `http://YOUR_PC_IP:5000` |

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Dev UI with hot reload (needs Flask on :5000) |
| `npm run build` | Build into `dist/` for Flask to serve |
| `npm run preview` | Preview production build (proxy to Flask) |

## Features

- Upload leaf photo → AI diagnosis
- Clarification Q&A when confidence is low
- Scan history (stored on server in `prediction_history.json`)
- Light / dark theme
- No login required
