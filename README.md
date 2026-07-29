# TDS Project 1 — Data Analyst Telegram Bot

An LLM agent exposed as a Telegram bot. When messaged a data-analysis question,
it works out the answer (writing and running Python code) and replies with a
**single JSON object**:

```json
{"answer": <the answer, in the shape the question asked for>, "log_url": "https://storage.googleapis.com/abhi-iitm-tds-bot-logs/run.jsonl"}
```

- **Bot username:** `@abhi_data_analyst_2026_bot`
- **Student:** `23f1002539@ds.study.iitm.ac.in`
- **LLM:** OpenRouter (`google/gemini-2.5-flash` by default) — any OpenAI-compatible model works.
- **Run log:** appended to `run.jsonl` and mirrored to a public GCS bucket so the grader can `wget` `log_url`.

## How it works

```
Telegram getUpdates (long-poll) ──► per-chat FIFO worker ──► agent.run_agent()
                                                          │
                                   sandbox.run_python() ◄─┘  (pandas/requests/bs4)
                                                          │
                                   FINAL_ANSWER: <json>  ◄─┘
                                                          │
                          {"answer": …, "log_url": …} ───► Telegram sendMessage
                                                          └► logs.append() ──► GCS
```

- **Per-chat FIFO:** messages within one conversation are handled strictly in
  order (the grader alternates send→wait-for-reply); different chats run in
  parallel (the grader runs ~5 concurrently).
- **Multi-turn:** context-only turns get a quick `{"ack": true}`; the turn that
  asks for an answer runs the full agent. Only the bot's last reply is graded.
- **Agent protocol (model-agnostic):** the model emits fenced ```python blocks
  (executed in a subprocess) and finishes with `FINAL_ANSWER: <json>`. No native
  function-calling, so it works on any OpenRouter model.

## Files

| file | purpose |
|------|---------|
| `bot.py` | Telegram long-polling loop, per-chat workers, reply formatting |
| `agent.py` | ReAct loop over the OpenAI-compatible endpoint |
| `sandbox.py` | Subprocess Python execution with timeout |
| `logs.py` | JSONL run log + public GCS mirror |
| `.env` | secrets (gitignored) |
| `Dockerfile` | container image for GCP |

## Run locally

```bash
uv sync                                   # or: pip install -r requirements.txt
cp .env.example .env                      # then fill in secrets
uv run python bot.py                      # or: python bot.py
```

For the GCS log mirror to work locally, authenticate application-default creds:
```bash
gcloud auth application-default login
```
(On a GCP VM / Cloud Run this is automatic.)

## Test the agent directly (no Telegram)

```python
import agent, os
from dotenv import load_dotenv; load_dotenv()
r = agent.run_agent(
    'Which of these is largest? 12, 7, 19, 3. Reply with ONLY this JSON object and nothing else: {"answer": {"largest": <number>}, "log_url": "<url>"}',
    [],
    model=os.environ["OPENROUTER_MODEL"],
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url=os.environ["OPENROUTER_BASE_URL"],
)
print(r["answer"])   # -> {'largest': 19}
```

## Test end-to-end with the public grading pipeline

```bash
git clone https://github.com/Jivraj-18/tds-p1-t2-2026-telegram-bot
cd tds-p1-t2-2026-telegram-bot
# set TELEGRAM_API_ID / TELEGRAM_API_HASH / TELEGRAM_SESSION_STRING (see login.py)
# put your own questions in evals/questions.json, your bot in students.csv
python collect.py   # sends messages to your bot and records replies
python grade.py     # exact-matches the last reply against the answer key
```

## Deploy on GCP (always-on, recommended: Compute Engine VM)

The bot uses outbound long-polling — no inbound port needed for Telegram.
Use an always-on VM (not serverless-to-zero) so the bot stays reachable during
grading.

```bash
# on a fresh e2-medium VM (Debian), as needed:
sudo apt update && sudo apt install -y python3-venv git
git clone https://github.com/23f1002539/tds-data-analyst-bot && cd tds-data-analyst-bot
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
cp .env.example .env && nano .env        # fill secrets
# ADC for the GCS log mirror is automatic on a VM with a service account that
# has Storage Object Admin on the logs bucket.
nohup python bot.py > bot.log 2>&1 &
```

Or with Docker:
```bash
docker build -t tds-bot .
docker run -d --restart unless-stopped --env-file .env tds-bot
```

A systemd unit (`tds-bot.service`) is recommended for auto-restart:

```ini
# /etc/systemd/system/tds-bot.service
[Unit]
Description=TDS Data Analyst Telegram Bot
After=network-online.target

[Service]
WorkingDirectory=/opt/tds-data-analyst-bot
ExecStart=/opt/tds-data-analyst-bot/.venv/bin/python bot.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

The VM's service account needs **Storage Object Admin** on
`gs://abhi-iitm-tds-bot-logs` (for the public run log) and the logs bucket must
be public (`roles/storage.objectViewer` for `allUsers`) so the grader can fetch
`log_url`.