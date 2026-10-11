"""
Step 2: Replicate the paper's proof-of-concept age gate results with your rebuilt gate.

Needs (in the same folder):
  gate.py              (Step 1)
  system_prompt.txt    (Table 7 of the paper)
  prompts_paper.csv    (the paper's prompt library, Tables 4-6, age-stylized column)

prompts_paper.csv format (one prompt per row):
  group,type,text
  group = 5, 7, 9, 11, teen, adult
  type  = explicit, implicit, age_check, action_check

Run:
  py replicate.py

Results are saved to results/replication.jsonl as the run goes.
If it stops (rate limit, daily limit, crash), just run it again: finished tests are skipped.
"""

import csv
import json
import os
import time
from collections import defaultdict

import httpx
from google.genai import errors

from gate import client, CONFIG, MODEL

REPEATS = 3
DELAY_SECONDS = float(os.environ.get("DELAY_SECONDS", "7"))  # pause between messages (free tier rate limit)
PROMPTS_FILE = "prompts_paper.csv"
RESULTS_FILE = os.path.join("results", f"replication_{MODEL.replace('/', '_')}.jsonl")  # one file per model, never mixed
CHILD_GROUPS = ["5", "7", "9", "11"]
ALL_GROUPS = CHILD_GROUPS + ["teen", "adult"]


# ---------- load prompts ----------
def load_prompts():
    prompts = defaultdict(lambda: defaultdict(list))
    with open(PROMPTS_FILE, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            group = row["group"].strip().lower()
            ptype = row["type"].strip().lower()
            text = row["text"].strip()
            if text:
                prompts[group][ptype].append(text)
    for g in ALL_GROUPS:
        for needed in ("age_check", "action_check"):
            if not prompts[g][needed]:
                raise SystemExit(f"Missing '{needed}' row for group {g} in {PROMPTS_FILE}")
    return prompts


# ---------- talking to the gate ----------
def send(chat, msg):
    """Send one message with a pause before it, retrying on rate limits or server errors."""
    for attempt in range(6):
        time.sleep(DELAY_SECONDS)
        try:
            return chat.send_message(msg)
        except errors.APIError as e:
            code = getattr(e, "code", None)
            text = str(e)
            if code == 429 and "PerDay" in text:
                print("\n    DAILY free-tier limit reached for this model. Full message from Google:\n")
                print(text)
                raise SystemExit("\nStopped. Your results so far are saved; run again after the daily limit resets.")
            if code in (429, 500, 503):
                wait = 30 * (attempt + 1)
                print(f"    API busy or rate limited ({code}), waiting {wait}s...")
                if attempt == 0:
                    print(f"    Google says: {text[:500]}")
                time.sleep(wait)
                continue
            raise
        except (httpx.HTTPError, ConnectionError, OSError) as e:
            # Internet hiccup (e.g. "connection was forcibly closed"): wait and try again.
            wait = 20 * (attempt + 1)
            print(f"    Network problem ({type(e).__name__}), retrying in {wait}s...")
            time.sleep(wait)
            continue
    raise RuntimeError("Too many retries, try again later.")


def run_chat(messages, roles):
    """Run one chat. Stops as soon as the gate calls end_chat.
    roles marks each message as 'prompt', 'age_check' or 'action_check'."""
    chat = client.chats.create(model=MODEL, config=CONFIG)
    transcript = []
    age_checks_sent = 0
    for i, (msg, role) in enumerate(zip(messages, roles)):
        if role == "age_check":
            age_checks_sent += 1
        resp = send(chat, msg)
        if resp.function_calls:
            transcript.append({"user": msg, "role": role, "blocked": True,
                               "args": dict(resp.function_calls[0].args or {})})
            return {"blocked": True, "blocked_at_turn": i,
                    "age_checks_before_block": age_checks_sent, "transcript": transcript}
        transcript.append({"user": msg, "role": role, "bot": resp.text})
    return {"blocked": False, "blocked_at_turn": None,
            "age_checks_before_block": None, "transcript": transcript}


# ---------- build the experiments exactly like the paper ----------
def build_experiments(prompts):
    exps = []
    for g in ALL_GROUPS:
        age_check = prompts[g]["age_check"][0]
        action_check = prompts[g]["action_check"][0]

        # Individual: prompt -> age check -> action check
        for ptype in ("explicit", "implicit"):
            for idx, p in enumerate(prompts[g][ptype]):
                exps.append({
                    "id": f"ind-{g}-{ptype}-{idx}",
                    "kind": "individual", "group": g, "type": ptype,
                    "messages": [p, age_check, action_check],
                    "roles": ["prompt", "age_check", "action_check"],
                })

        # Sequential: all implicit (each + age check), then all explicit (each + age check), then action check
        msgs, roles = [], []
        for p in prompts[g]["implicit"] + prompts[g]["explicit"]:
            msgs += [p, age_check]
            roles += ["prompt", "age_check"]
        msgs.append(action_check)
        roles.append("action_check")
        exps.append({"id": f"seq-{g}", "kind": "sequential", "group": g, "type": "sequential",
                     "messages": msgs, "roles": roles})
    return exps


# ---------- saving and resuming ----------
def load_done():
    done = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done[(r["id"], r["repeat"])] = r
    return done


def save(record):
    os.makedirs("results", exist_ok=True)
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------- summary vs the paper ----------
def summarize(done):
    rows = list(done.values())

    def rate(filter_fn):
        sel = [r for r in rows if filter_fn(r)]
        blocked = sum(r["blocked"] for r in sel)
        return blocked, len(sel)

    print("\n================ RESULTS vs PAPER ================")
    print(f"Model: {MODEL}\n")

    b, n = rate(lambda r: r["kind"] == "individual" and r["group"] in CHILD_GROUPS and r["type"] == "explicit")
    print(f"Kids stating age (individual):     blocked {b}/{n}   | paper: 59/60")

    b, n = rate(lambda r: r["kind"] == "individual" and r["group"] in CHILD_GROUPS and r["type"] == "implicit")
    print(f"Kids giving hints (individual):    blocked {b}/{n}   | paper: 6/72 (all age 5)")
    for g in CHILD_GROUPS:
        b, n = rate(lambda r, g=g: r["kind"] == "individual" and r["group"] == g and r["type"] == "implicit")
        print(f"    age {g:>2} hints:                   blocked {b}/{n}")

    seq = [r for r in rows if r["kind"] == "sequential" and r["group"] in CHILD_GROUPS]
    before4 = sum(1 for r in seq if r["blocked"] and r["age_checks_before_block"] is not None
                  and r["age_checks_before_block"] < 4)
    print(f"Kids, long chats (sequential):     blocked before 4th age check {before4}/{len(seq)} | paper: all")

    b, n = rate(lambda r: r["group"] in ("teen", "adult"))
    print(f"Teens and adults (all tests):      blocked {b}/{n}   | paper: expected ~0")
    print("==================================================")


def main():
    prompts = load_prompts()
    exps = build_experiments(prompts)
    done = load_done()
    total = len(exps) * REPEATS
    count = len(done)
    print(f"{len(exps)} experiments x {REPEATS} repeats = {total} runs ({count} already done)")

    for rep in range(1, REPEATS + 1):
        for e in exps:
            if (e["id"], rep) in done:
                continue
            count += 1
            print(f"[{count}/{total}] {e['id']} repeat {rep}")
            result = run_chat(e["messages"], e["roles"])
            record = {"id": e["id"], "repeat": rep, "kind": e["kind"], "group": e["group"],
                      "type": e["type"], "model": MODEL, **result}
            save(record)
            done[(e["id"], rep)] = record
            print(f"    blocked={result['blocked']}")

    summarize(done)


if __name__ == "__main__":
    main()
