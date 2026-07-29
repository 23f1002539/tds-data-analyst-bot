"""Run logging — one JSON object per line (JSONL).

Every handled message appends a line to a local run.jsonl and re-uploads the
file to a public GCS bucket so the grader can wget log_url. GCS auth uses
Application Default Credentials (automatic on GCP VMs/Cloud Run, or via
`gcloud auth application-default login` locally). If no bucket is configured,
we just keep the local file and serve it over HTTP from the bot.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone

_LOCK = threading.Lock()
_LOCAL_PATH = os.environ.get("RUN_LOG_PATH", "run.jsonl")
GCS_BUCKET = os.environ.get("GCS_LOGS_BUCKET", "")
GCS_OBJECT = os.environ.get("GCS_LOGS_OBJECT", "run.jsonl")
LOG_URL = os.environ.get(
    "LOG_URL",
    f"https://storage.googleapis.com/{GCS_BUCKET}/{GCS_OBJECT}" if GCS_BUCKET else "",
)


def append(entry: dict) -> None:
    entry = {**entry, "ts": datetime.now(timezone.utc).isoformat()}
    line = json.dumps(entry, ensure_ascii=False, default=str)
    with _LOCK:
        with open(_LOCAL_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        _upload_to_gcs()


def _upload_to_gcs() -> None:
    if not GCS_BUCKET:
        return
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(GCS_OBJECT)
        blob.cache_control = "no-cache"
        blob.upload_from_filename(_LOCAL_PATH, content_type="application/x-ndjson")
        # make this object publicly readable (idempotent)
        try:
            blob.make_public()
        except Exception:
            pass
    except Exception as e:
        # logging must never break the bot
        print(f"[logs] GCS upload failed: {e}", flush=True)


def local_log_url(public_host: str) -> str:
    """Fallback log_url when no GCS bucket is configured."""
    if GCS_BUCKET:
        return LOG_URL
    return f"{public_host.rstrip('/')}/run.jsonl"