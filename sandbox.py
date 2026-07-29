"""Sandboxed Python execution for the data-analyst agent.

The agent emits Python code; we run it in a subprocess using the same
interpreter/env as the bot (so pandas, requests, bs4, etc. are available),
capture stdout/stderr, and return a truncated result. Code is executed in a
fresh temporary working directory with a hard timeout so a runaway script
can't stall the bot.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

MAX_OUTPUT = 6000          # chars of stdout+stderr returned to the LLM
DEFAULT_TIMEOUT = 45       # seconds per code execution


def run_python(code: str, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Execute *code* in a subprocess. Returns {ok, stdout, stderr, returncode}."""
    code = textwrap.dedent(code)
    workdir = tempfile.mkdtemp(prefix="sandbox_")
    script = os.path.join(workdir, "agent_code.py")
    try:
        with open(script, "w") as f:
            f.write(
                "# auto-generated agent code\n"
                "import sys, json, math, statistics, csv, re, itertools, collections\n"
                "try:\n"
                "    import requests, pandas as pd, numpy as np\n"
                "except Exception as _e:\n"
                f"    print('IMPORT_WARN:', _e, file=sys.stderr)\n"
                + code
            )
        proc = subprocess.run(
            [sys.executable, "-I", script],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        out = (proc.stdout or "")
        err = (proc.stderr or "")
        combined = out + ("\n" + err if err else "")
        if len(combined) > MAX_OUTPUT:
            combined = combined[:MAX_OUTPUT] + f"\n...[truncated {len(combined)-MAX_OUTPUT} chars]"
        return {
            "ok": proc.returncode == 0,
            "stdout": out[:MAX_OUTPUT],
            "stderr": err[:MAX_OUTPUT],
            "returncode": proc.returncode,
            "combined": combined,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "stdout": "", "stderr": f"TIMEOUT after {timeout}s", "returncode": -1,
                "combined": f"TIMEOUT after {timeout}s"}
    except Exception as e:
        return {"ok": False, "stdout": "", "stderr": str(e), "returncode": -1, "combined": str(e)}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)