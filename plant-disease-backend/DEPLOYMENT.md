# Deployment Guide — Plant Disease Backend + Flutter App

## Quick start (local / university demo)

### 1. Backend

```powershell
cd "d:\ai data\Final\plant-disease-backend"
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` — **minimum required:**

```env
JWT_SECRET=your-long-random-secret-at-least-32-characters
```

Optional (cloud backup for history):

```env
SUPABASE_URL=https://YOUR_PROJECT.supabase.co
SUPABASE_KEY=your_anon_or_service_key
```

Start server:

```powershell
python app.py
```

You should see: `History: local JSON always; Supabase backup=on/off`

### 2. Flutter app

In `project/lib/utils/constants.dart`, set your PC IPv4:

```dart
static const String pcLanIpForPhysicalDevice = '192.168.x.x';
```

```powershell
cd "d:\ai data\Final\project"
flutter run
```

Phone and PC must be on the **same Wi‑Fi**. Windows Firewall must allow Python on port **5000**.

---

## Scan history (fixed)

| Storage | File / service | When used |
|---------|----------------|-----------|
| **Primary** | `prediction_history.json` | Always — works without Supabase |
| **Optional** | Supabase `predictions` table | Backup if URL/key set and table exists |

- **GET** `/history` — list scans (JWT required)
- **POST** `/history` — save refined result after farmer Q&A (JWT required)
- **POST** `/predict` — auto-saves each scan

---

## Optional Supabase setup

1. Create project at [supabase.com](https://supabase.com)
2. Run SQL from `supabase_predictions.sql` in SQL Editor
3. Add `SUPABASE_URL` and `SUPABASE_KEY` to `.env`

If Supabase fails, history still works via local JSON.

---

## Production checklist

- [ ] Set strong `JWT_SECRET` in `.env`
- [ ] Set `SUPABASE_AUTH_ENABLED=false` unless you configured Supabase email
- [ ] Copy `models/model.pkl`, `scaler.pkl`, `label_encoder.pkl` into `models/`
- [ ] Run `python app.py` or `waitress-serve` via `wsgi.py`
- [ ] Update Flutter `pcLanIpForPhysicalDevice` to server IP
- [ ] Register/login in app, run a scan, open **Scan History** tab
- [ ] Confirm `prediction_history.json` grows after scans

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| History empty | Log in again; ensure backend restarted after update |
| 401 on history | Token expired — sign out and sign in |
| App cannot reach API | Check IP in `constants.dart`, backend running, firewall |
| Supabase errors in log | OK if local history works; fix SQL or ignore Supabase |

---

## Files

| File | Role |
|------|------|
| `history_store.py` | Local JSON history |
| `auth.py` | `save_prediction` / `get_history` |
| `prediction_history.json` | Created automatically at runtime |
| `supabase_predictions.sql` | Optional cloud table |
