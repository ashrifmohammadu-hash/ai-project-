"""Simple alerting helpers (placeholder).

This module contains a small helper for flood alerts. Replace with
real notification integration (SMS/Email/Push) when available.
"""
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def send_flood_alert(area_name: str, level: int, message: str = None) -> bool:
    """Log a flood alert; return True if the alert was 'sent'.

    This is a safe placeholder that records the alert to logs. Calling
    code should replace this with a real notifier.
    """
    payload = {
        "area": area_name,
        "level": level,
        "message": message or f"Flood alert level {level} in {area_name}",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    logger.warning("Flood alert: %s", payload)
    return True
