import os
from dotenv import load_dotenv
import agent

load_dotenv()

q = ('Here are monthly sales for three regions: North=120, South=85, East=190, West=60. '
     'Reply with ONLY this JSON object and nothing else: '
     '{"answer": {"region": "<region name with the highest sales>"}, "log_url": "<url>"}')

r = agent.run_agent(
    q, [],
    model=os.environ["OPENROUTER_MODEL"],
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url=os.environ["OPENROUTER_BASE_URL"],
)
print("ANSWER:", r["answer"])
print("RAW:", r["raw"])
print("ERROR:", r["error"])
print("ITERS:", len(r["trace"]))
for i, t in enumerate(r["trace"]):
    k = "tool" if "tool" in t else "assistant"
    print(f"--- trace {i} ({k}) ---")
    print((t.get("combined") or t.get("assistant") or "")[:500])