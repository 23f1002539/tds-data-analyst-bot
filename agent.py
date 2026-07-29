"""The data-analyst agent.

A text-based ReAct loop over an OpenAI-compatible chat endpoint (OpenRouter by
default). The model computes by emitting fenced ```python blocks (executed by
sandbox.run_python) and finishes by emitting a line `FINAL_ANSWER: <json>`.
We deliberately avoid native function-calling so the same loop works across
every OpenRouter model (GPT, Gemini, Claude, ...).

The agent returns only the inner *answer* value (the exact JSON shape the
question asked for). The bot wraps it as {"answer": ..., "log_url": ...}.
"""
from __future__ import annotations

import json
import os
import re
import requests

import sandbox

SYSTEM = """You are an expert data analyst. You receive a data-analysis question and must
work out the answer, then reply with ONLY the answer in the exact JSON shape the
question requests.

HOW TO WORK:
- To compute anything, output a fenced code block:
  ```python
  # your code
  ```
  You will see its stdout and stderr. You may run code several times.
- You have pandas (as pd), numpy (as np), requests, bs4, and the stdlib.
- If the question points at a public dataset or URL, fetch it with requests and
  parse it. If the data is inline in the message, parse it directly.
- Numbers must be computed, not guessed. Round only if the question asks.

HOW TO FINISH:
- When you have the answer, output exactly one line:
  FINAL_ANSWER: <a single JSON value matching the requested shape>
- That JSON value is the contents of the "answer" field the user asked for.
  Do NOT include the keys "answer" or "log_url" yourself, and do NOT wrap it.
  Examples of correct FINAL_ANSWER lines:
    FINAL_ANSWER: {"state": "Assam"}
    FINAL_ANSWER: 42
    FINAL_ANSWER: [3, 1, 4]
    FINAL_ANSWER: "2023-04"
- Output nothing after the FINAL_ANSWER line. No explanation, no prose.

SHAPE FIDELITY (critical — answers are exact-matched):
- Use EXACTLY the keys the question's JSON template shows — no more, no fewer.
  Copy the key names and structure from the template the message provides.
- NEVER add extra keys, even if informative. If the template is
  {"state": "<state name>"} output {"state": "Assam"} — NOT
  {"state": "Assam", "reduction": 58}. Extra keys fail grading.
- Match value types: a string placeholder "<state name>" -> a string;
  a number placeholder <number> -> a number; a list -> a list. Round numbers
  only if the question asks for rounding.
- If no template is given, return the minimal JSON that answers the question.

Be rigorous. Verify the shape matches what the question asked before finishing."""

MAX_ITER = 6


def _chat(model, messages, api_key, base_url):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 3000,
    }
    # reasoning models benefit from a low effort hint (cheaper, faster)
    if any(k in model for k in ("gpt-5", "gemini", "o1", "o3")):
        payload["reasoning"] = {"effort": "low"}
    r = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=180,
    )
    data = r.json()
    if "choices" not in data:
        raise RuntimeError(f"LLM error: {data}")
    return data["choices"][0]["message"]["content"] or ""


_PYTHON_BLOCK = re.compile(r"```python\s*\n(.*?)```", re.S)
_FINAL = re.compile(r"FINAL_ANSWER:\s*(.+)", re.S)


def _extract_final(text):
    """Return (answer_value, raw_json_str) if a FINAL_ANSWER is present, else (None, None)."""
    m = _FINAL.search(text)
    if not m:
        return None, None
    raw = m.group(1).strip().splitlines()[0].strip()
    # strip a trailing markdown fence or stray quote if present
    raw = raw.rstrip("`")
    try:
        return json.loads(raw), raw
    except json.JSONDecodeError:
        # try to grab the first balanced JSON value
        for cand in re.findall(r"(\{.*\}|\[.*\]|true|false|null|-?\d+(?:\.\d+)?|\".*\")", raw, re.S):
            try:
                return json.loads(cand), cand
            except json.JSONDecodeError:
                continue
        return None, raw


def _extract_template(question):
    """Find the JSON answer template the question asks for.

    The spec format is {"answer": <template>, "log_url": "..."}. We scan for
    balanced {...} substrings, and return <template> from the first object that
    has an "answer" key. Returns None if no such object is found (then we leave
    the answer untouched — never filter on a guessed template).
    """
    candidates = []
    depth = 0
    start = None
    in_str = False
    esc = False
    for i, ch in enumerate(question):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(question[start:i + 1])
                start = None
    for c in candidates:
        try:
            o = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(o, dict) and "answer" in o:
            return o["answer"]
    return None


def _enforce_shape(answer, template):
    """Filter `answer` to the keys/structure of `template` (best-effort, recursive).

    Guarantees no extra keys leak into the graded answer (the grader
    exact-matches). If template is None, returns answer unchanged.
    """
    if template is None or answer is None:
        return answer
    if isinstance(template, dict) and isinstance(answer, dict):
        out = {}
        for k, tv in template.items():
            if k in answer:
                out[k] = _enforce_shape(answer[k], tv)
        return out
    if isinstance(template, list) and isinstance(answer, list):
        if template and isinstance(template[0], dict):
            return [_enforce_shape(a, template[0]) for a in answer]
        return answer
    return answer


def run_agent(question, history, *, model, api_key, base_url, on_step=None):
    """Run the agent. Returns dict {answer, raw, trace, error}.

    *question* is the latest user message; *history* is a list of prior
    (role, text) turns in the same chat for multi-turn context.
    """
    ctx = ""
    if history:
        ctx = "Earlier messages in this conversation (for context):\n" + "\n".join(
            f"- {r}: {t}" for r, t in history
        ) + "\n\n"
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": ctx + "Current message:\n" + question},
    ]
    trace = []
    for i in range(MAX_ITER):
        resp = _chat(model, messages, api_key, base_url)
        trace.append({"iter": i, "assistant": resp})
        if on_step:
            on_step(resp)

        ans, raw = _extract_final(resp)
        if ans is not None or raw:
            return {"answer": _enforce_shape(ans, _extract_template(question)),
                    "raw": raw, "trace": trace, "error": None}

        blocks = _PYTHON_BLOCK.findall(resp)
        if blocks:
            code = blocks[-1]
            result = sandbox.run_python(code)
            trace.append({"iter": i, "tool": "run_python", "result": result["combined"]})
            messages.append({"role": "assistant", "content": resp})
            messages.append({
                "role": "user",
                "content": (
                    f"Code output (returncode={result['returncode']}):\n"
                    f"--- stdout ---\n{result['stdout']}\n"
                    f"--- stderr ---\n{result['stderr']}\n"
                    "Continue. If you now have the answer, output "
                    "FINAL_ANSWER: <json>. Otherwise output another python block."
                ),
            })
            continue

        # No code and no final answer: maybe it blurted JSON directly
        stripped = resp.strip()
        try:
            return {"answer": _enforce_shape(json.loads(stripped), _extract_template(question)),
                    "raw": stripped, "trace": trace, "error": None}
        except json.JSONDecodeError:
            pass
        messages.append({"role": "assistant", "content": resp})
        messages.append({
            "role": "user",
            "content": "You did not output a python block or a FINAL_ANSWER line. Either emit a "
                       "```python block to compute the answer, or finish with "
                       "FINAL_ANSWER: <json value in the requested shape>.",
        })

    return {"answer": None, "raw": None, "trace": trace,
            "error": "max iterations reached without a final answer"}