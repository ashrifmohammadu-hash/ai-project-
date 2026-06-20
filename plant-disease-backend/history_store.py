"""Local prediction history (always available for deployment)."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

HISTORY_FILE = Path(__file__).parent / "prediction_history.json"
MAX_PER_USER = 100


def _load_all() -> Dict[str, List[Dict[str, Any]]]:
    if not HISTORY_FILE.exists():
        return {}
    try:
        raw = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return raw
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read history file: %s", exc)
    return {}


def _save_all(data: Dict[str, List[Dict[str, Any]]]) -> None:
    HISTORY_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def append_prediction(user_id: str, record: Dict[str, Any]) -> Dict[str, Any]:
    """Append one scan record for a user (newest first)."""
    data = _load_all()
    rows = data.setdefault(user_id, [])

    entry = {
        "id": record.get("id") or str(uuid.uuid4()),
        "user_id": user_id,
        "disease": record.get("disease") or "Unknown",
        "display_name": record.get("display_name"),
        "plant_type": record.get("plant_type"),
        "confidence": float(record.get("confidence") or 0),
        "image_name": record.get("image_name") or "",
        "created_at": record.get("created_at")
        or datetime.now(timezone.utc).isoformat(),
        "is_confident": bool(record.get("is_confident", True)),
        "needs_clarification": bool(record.get("needs_clarification", False)),
    }
    if record.get("treatment"):
        entry["treatment"] = record["treatment"]

    rows.insert(0, entry)
    data[user_id] = rows[:MAX_PER_USER]
    _save_all(data)
    logger.info("History saved locally for user %s: %s", user_id, entry["disease"])
    return entry


def list_predictions(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Return newest predictions for a user."""
    rows = _load_all().get(user_id, [])
    return rows[:limit]
