"""
Prints the gate's full admin report for
  (1) every block where the gate's own age estimate was 13 or older, and
  (2) every teen/adult who got blocked,
so you can check each one by eye.

Run:  py check_contradictions.py      (uses no requests)
"""
import json
import os

from gate import MODEL
from robustness import estimate_when_blocked

safe = MODEL.replace("/", "_")
for name in (f"replication_{safe}.jsonl", f"robustness_{safe}.jsonl"):
    path = os.path.join("results", name)
    if not os.path.exists(path):
        continue
    print(f"\n######## {name} ########")
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not r["blocked"]:
                continue
            est = estimate_when_blocked(r)
            adult = r["group"] in ("teen", "adult", "innocent_adult")
            if not adult and (est is None or est < 13):
                continue
            last = r["transcript"][-1]
            tag = "ADULT/TEEN BLOCKED" if adult else "CONTRADICTION"
            print(f"\n--- [{tag}] {r['id']} repeat {r['repeat']} (gate's estimate: {est}) ---")
            print("FIRST MESSAGE :", r["transcript"][0]["user"])
            print("BLOCKED AT    :", last.get("user"))
            print("ADMIN REPORT  :", last.get("args", {}).get("admin_report"))
