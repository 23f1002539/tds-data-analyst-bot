#!/usr/bin/env python3
"""TDS Project 1 — Data Analyst Telegram bot.

Receives plain-text data-analysis questions via Telegram long-polling, runs an
LLM agent (with Python code execution) to solve them, and replies with a single
JSON object: {"answer": <shape the question asked for>, "log_url": <public URL>}.

Multi-turn: the grader sends a short sequence of messages, waiting for a reply
after each. We reply with a quick {"ack": true} to context-only messages and
with the real answer to the message that actually asks for one. Only the bot's
last reply is graded.

Per-chat FIFO workers keep one conversation's messages strictly ordered while
different chats are handled in parallel (the grader runs ~5 concurrently).
"""
from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
from collections import defaultdict

from dotenv import load_dotenv

load_dotenv()  # must run before `import logs`, which reads GCS_LOGS_BUCKET at import time

import requests

import agent
import logs

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
EMAIL = os.environ.get("EMAIL", "23f1002539@ds.study.iitm.ac.in")
LOG_URL = logs.LOG_URL

TG_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# --- conversation state ------------------------------------------------------
_chat_history: dict[int, list[tuple[str, str]]] = defaultdict(list)
_chat_queues: dict[int, queue.Queue] = {}
_chat_workers: dict[int, threading.Thread] = {}

# messages that actually ask for an answer (the rest get a quick ack)
_Q_RE = re.compile(
    r"reply with|respond with|return only|reply with only|answer with|"
    r'"answer"|log_url|reply only|respond only',
    re.IGNORECASE,
)


def looks_like_question(text: str) -> bool:
    return bool(_Q_RE.search(text))


def tg_send(chat_id: int, text: str) -> None:
    requests.post(
        f"{TG_API}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=30,
    )


def handle(chat_id: int, text: str) -> None:
    hist = _chat_history[chat_id]
    hist.append(("user", text))

    if not looks_like_question(text):
        # context-only turn: acknowledge fast so the grader can send the next
        reply = json.dumps({"ack": True}, separators=(",", ":"))
        hist.append(("assistant", reply))
        tg_send(chat_id, reply)
        logs.append({
            "email": EMAIL, "chat_id": chat_id, "kind": "ack",
            "incoming": text[:2000], "reply": reply,
        })
        return

    t0 = time.time()
    result = agent.run_agent(
        text,
        hist[:-1],
        model=OPENROUTER_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )
    answer = result["answer"]
    if answer is None and result["raw"]:
        answer = result["raw"]  # fall back to the raw string the agent produced
    reply = json.dumps({"answer": answer, "log_url": LOG_URL}, separators=(",", ":"))
    hist.append(("assistant", reply))
    tg_send(chat_id, reply)
    logs.append({
        "email": EMAIL,
        "chat_id": chat_id,
        "kind": "answer",
        "model": OPENROUTER_MODEL,
        "incoming": text[:4000],
        "history_turns": len(hist) - 1,
        "answer": answer,
        "raw": result["raw"],
        "error": result["error"],
        "iters": len(result["trace"]),
        "trace": result["trace"],
        "elapsed_s": round(time.time() - t0, 2),
        "reply": reply,
    })


def chat_worker(chat_id: int, q: queue.Queue) -> None:
    while True:
        text = q.get()
        try:
            handle(chat_id, text)
        except Exception as e:
            print(f"[worker chat={chat_id}] error: {e}", flush=True)
            try:
                tg_send(chat_id, json.dumps({"answer": None, "log_url": LOG_URL}))
            except Exception:
                pass
        finally:
            q.task_done()


def dispatch(chat_id: int, text: str) -> None:
    if chat_id not in _chat_workers:
        q: queue.Queue = queue.Queue()
        _chat_queues[chat_id] = q
        t = threading.Thread(target=chat_worker, args=(chat_id, q), daemon=True)
        _chat_workers[chat_id] = t
        t.start()
    _chat_queues[chat_id].put(text)


def poll() -> None:
    print(f"[bot] polling as model={OPENROUTER_MODEL} log_url={LOG_URL}", flush=True)
    offset = 0
    while True:
        try:
            r = requests.get(
                f"{TG_API}/getUpdates",
                params={"offset": offset, "timeout": 60},
                timeout=70,
            )
            for update in r.json().get("result", []):
                offset = update["update_id"] + 1
                msg = update.get("message") or update.get("edited_message")
                if not msg:
                    continue
                chat = msg.get("chat", {})
                text = msg.get("text", "")
                if text and not text.startswith("/"):
                    print(f"[bot] <- chat={chat.get('id')} text={text[:80]!r}", flush=True)
                    dispatch(chat["id"], text)
                elif text.startswith("/"):
                    dispatch(chat["id"], text)  # let handler see commands too
        except Exception as e:
            print(f"[bot] poll error: {e}", flush=True)
            time.sleep(3)


if __name__ == "__main__":
    poll()