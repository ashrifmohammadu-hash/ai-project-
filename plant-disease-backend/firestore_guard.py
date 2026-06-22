"""
Lightweight Firestore guard utility.
Provides a safe `get_firestore_client()` that returns None when Firestore
is not configured, avoiding crashes when code expects a client.
"""
import os
import logging

logger = logging.getLogger(__name__)


def get_firestore_client():
    """Return a Firestore client if configured, otherwise None.

    Looks for environment variable `FIRESTORE_PROJECT`. If not set, logs
    a warning and returns None so callers can fall back.
    """
    project = os.getenv("FIRESTORE_PROJECT")
    if not project:
        logger.warning("Firestore not configured: FIRESTORE_PROJECT not set")
        return None

    try:
        from google.cloud import firestore
        client = firestore.Client(project=project)
        return client
    except Exception as e:
        logger.warning("Failed to create Firestore client: %s", e)
        return None
