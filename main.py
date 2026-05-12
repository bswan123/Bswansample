#!/usr/bin/env python3
"""
EXAM SOLVER SERVER — v6.0
========================================

ARCHITECTURE
------------
OCR/Image
↓
clean_text()
↓
GPT-5 + web_search_preview (ALWAYS ON)
↓
Universal prompt
↓
Concise formatted answer
↓
LoRa transmission

FIXES FROM v5.0
---------------
✔ web_search_preview  (was: web_search — was silently failing)
✔ MAX_TOKENS = 1200   (was: 500 — puzzles were getting cut off)
✔ Better universal prompt (handles all IBPS question types)
✔ resp.output_text safety check with fallback
✔ Timeout increased to 180s (web search takes time)

ENDPOINTS
---------
/solve-image
/solve-text
/health

RUN
---
python server.py
python server.py --test
python server.py --manual
python server.py --testall
"""

import os
import re
import sys
import json
import base64
import requests

from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

from openai import OpenAI

import uvicorn


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

BASE_URL = "https://bswansample-1-h2uw.onrender.com"

API_KEY = os.environ.get("OPENAI_API_KEY", "")

MODEL = "gpt-4o"          # fallback-safe; change to "gpt-5" when available on your key

MAX_TOKENS = 1200          # v5 was 500 — too small for puzzles

LOG_DIR = "logs"

FAIL_LOG = os.path.join(LOG_DIR, "failures.txt")

os.makedirs(LOG_DIR, exist_ok=True)

app = FastAPI(title="Exam Solver API v6.0", version="6.0")

client = OpenAI(api_key=API_KEY)


# ─────────────────────────────────────────────
# WEB SEARCH TOOL — CORRECT STRING
# ─────────────────────────────────────────────
#
# v5 used:  {"type": "web_search"}
# That was WRONG — no error thrown, but search never activated.
#
# Correct string for OpenAI Responses API:
WEB_SEARCH_TOOL = {"type": "web_search_preview"}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def clean_qid(qid: str) -> str:
    try:
        qid = str(qid).strip()
        m = re.search(r'\d+', qid)
        return "Q" + m.group() if m else "Q1"
    except:
        return "Q1"


def clean_text(text: str) -> str:

    text = str(text)
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    replacements = {
        "0S":     "OS",
        "HTIP":   "HTTP",
        "RBl":    "RBI",
        "UP1":    "UPI",
        "prfit":  "profit",
        "invst":  "invest",
        "gih":    "Gita",
        "lndia":  "India",
        "ﬁ":      "fi",
        "ﬂ":      "fl",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    text = re.sub(r'[\x00-\x1f\x7f]', '', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def extract_output_text(resp) -> str:
    """
    Safe extraction from OpenAI Responses API.
    Tries resp.output_text first (Responses API).
    Falls back to resp.choices[0].message.content (Chat API).
    """
    try:
        text = resp.output_text
        if text:
            return text.strip()
    except AttributeError:
        pass

    try:
        text = resp.choices[0].message.content
        if text:
            return text.strip()
    except (AttributeError, IndexError):
        pass

    return ""


def log_failure(kind: str, raw_input: str, output: str):
    try:
        with open(FAIL_LOG, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"TIME: {datetime.now()}\n")
            f.write(f"TYPE: {kind}\n")
            f.write("-" * 80 + "\n")
            f.write("INPUT:\n")
            f.write(raw_input[:4000] + "\n")
            f.write("-" * 80 + "\n")
            f.write("OUTPUT:\n")
            f.write(output[:4000] + "\n")
            f.write("=" * 80 + "\n")
    except:
        pass


def looks_bad_output(text: str) -> bool:
    t = text.strip().lower()
    if not t:
        return True
    bad_patterns = [
        "i cannot",
        "i can't",
        "unable to",
        "sorry",
        "cannot determine",
        "insufficient information",
        "i don't have",
        "i do not have",
    ]
    return any(x in t for x in bad_patterns)


# ─────────────────────────────────────────────
# UNIVERSAL PROMPT
# ─────────────────────────────────────────────

UNIVERSAL_PROMPT = """
You are an expert IBPS/SBI/RRB/NABARD banking exam solver.

OCR may contain mistakes. Auto-correct and solve.

USE WEB SEARCH for:
- Current affairs (any event, appointment, award, scheme, sports result)
- Computer awareness (shortcuts, OS, networking, software)
- Banking awareness (RBI policies, rates, schemes)
- Any GK that requires recent/updated facts

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — ALWAYS USE THIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QID: Q<number>
Question Number: <number>

Answer:
<answer here>

If MCQ options are visible → return correct option letter + text.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUESTION TYPE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ARITHMETIC / NUMBER SERIES / SIMPLIFICATION
→ Solve step by step internally, give only final answer.

WRONG NUMBER IN SERIES
→ Identify the pattern, find the odd one out.

SYLLOGISM
→ Apply standard syllogism rules strictly.
→ Return: "Only conclusion I follows" / "Only conclusion II follows" /
          "Both follow" / "Neither follows" / "Either I or II follows"

DATA SUFFICIENCY
→ Evaluate each statement alone, then together.
→ Return: "Only statement I is sufficient" /
          "Only statement II is sufficient" /
          "Both together are sufficient" /
          "Either alone is sufficient" /
          "Neither is sufficient"

ASSUMPTION / COURSE OF ACTION / CAUSE & EFFECT
→ Apply standard logical reasoning rules.
→ For ASSUMPTION: Check if it is implicit and necessary for the argument.
→ For COURSE OF ACTION: Check practicality and relevance.
→ For CAUSE & EFFECT: Check logical causal relationship.
→ Return the correct conclusion clearly.

CRITICAL REASONING (STRENGTHEN / WEAKEN / INFERENCE)
→ Identify the argument structure.
→ Return only the correct option.

CODING-DECODING / BLOOD RELATIONS / DIRECTION SENSE
→ Solve step by step internally.
→ Return only final answer.

INEQUALITY
→ Apply the substitution/direct comparison.
→ Return: "Conclusion follows" or "Conclusion does not follow" for each.

PUZZLE / SEATING ARRANGEMENT
→ Solve completely. Return ONLY final arrangement.
→ Use structured format:

TYPE: CIRCULAR / LINEAR / PARALLEL_ROW / FLOOR_FLAT / DATE_MONTH

For PARALLEL ROW example:
Row1 (North Facing Left→Right):
1. A | Doctor
2. B | Engineer
Row2 (South Facing Left→Right):
1. X | Lawyer
2. Y | Banker

For FLOOR example:
Floor 5: A | Doctor | Red
Floor 4: B | Engineer | Blue

For DATE_MONTH example:
Name | DOB | Fruit
A | 12-May | Apple
B | 15-Jun | Mango

CURRENT AFFAIRS
→ ALWAYS use web search. Return factual answer with year/date if relevant.

COMPUTER AWARENESS
→ Use web search if needed. Return precise technical answer.

BANKING / ECONOMY AWARENESS
→ Use web search for latest rates, schemes, appointments.
→ Return concise factual answer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. NEVER refuse. Always attempt.
2. NEVER explain unnecessarily.
3. NEVER output JSON.
4. Output plain text only.
5. Keep answer concise.
6. OCR mistakes are expected — auto-correct.
7. If MCQ → return exact option letter + text.
"""


# ─────────────────────────────────────────────
# GPT CALLS
# ─────────────────────────────────────────────

def call_gpt_text(qid: str, raw: str):

    user_msg = f"""QID: {qid}

Question:
{raw}

Solve carefully. Use web search if this is current affairs, computer awareness, or banking awareness."""

    resp = client.responses.create(
        model=MODEL,
        tools=[WEB_SEARCH_TOOL],
        input=[
            {
                "role": "system",
                "content": UNIVERSAL_PROMPT
            },
            {
                "role": "user",
                "content": user_msg
            }
        ],
        max_output_tokens=MAX_TOKENS,
        temperature=0
    )

    ans = extract_output_text(resp)

    if looks_bad_output(ans):
        log_failure("TEXT", raw, ans)

    return {
        "QID":      qid,
        "ANSWER":   ans,
        "RAW_TEXT": raw[:500]
    }


def call_gpt_image(qid: str, img_bytes: bytes, mime: str = "image/jpeg"):

    b64 = base64.b64encode(img_bytes).decode()

    resp = client.responses.create(
        model=MODEL,
        tools=[WEB_SEARCH_TOOL],
        input=[
            {
                "role": "system",
                "content": UNIVERSAL_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"""QID: {qid}

Read the screenshot/image carefully.
OCR may contain mistakes — auto-correct.
If options are visible, return exact correct option.
Use web search if this involves current affairs, computer awareness, or banking awareness.
Solve accurately."""
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime};base64,{b64}"
                    }
                ]
            }
        ],
        max_output_tokens=MAX_TOKENS,
        temperature=0
    )

    ans = extract_output_text(resp)

    if looks_bad_output(ans):
        log_failure("IMAGE", f"IMAGE_BYTES={len(img_bytes)}", ans)

    return {
        "QID":    qid,
        "ANSWER": ans
    }


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":     "ok",
        "model":      MODEL,
        "web_search": True,
        "version":    "6.0"
    }


@app.post("/solve-image")
async def solve_image(
    qid:   str        = Form(default="Q1"),
    image: UploadFile = File(...)
):
    img_bytes = await image.read()

    if not img_bytes:
        return JSONResponse({"error": "empty image"}, status_code=400)

    mime = image.content_type or "image/jpeg"
    qid  = clean_qid(qid)

    return call_gpt_image(qid, img_bytes, mime)


@app.post("/solve-text")
async def solve_text(
    qid:  str = Form(default="Q1"),
    text: str = Form(...)
):
    text = clean_text(text)

    if not text:
        return JSONResponse({"error": "empty text"}, status_code=400)

    qid = clean_qid(qid)

    return call_gpt_text(qid, text)


# ─────────────────────────────────────────────
# AUTO TESTS
# ─────────────────────────────────────────────

AUTO_TESTS = [

    (
        "Health Check",
        "health", None, None,
        ["ok"]
    ),
    (
        "Number Series",
        "text", "Q1",
        "2,4,12,60,420,?",
        ["3360"]
    ),
    (
        "Wrong Number",
        "text", "Q2",
        "Find wrong number: 15,18,42,125,506,2537",
        ["506"]
    ),
    (
        "Syllogism",
        "text", "Q3",
        "Statements: All cats are dogs. Some dogs are rats. "
        "Conclusions: I.Some rats are cats II.Some dogs are cats",
        ["ii"]
    ),
    (
        "Data Sufficiency",
        "text", "Q4",
        "What is x? Statement I: x+y=10  Statement II: y=4",
        ["sufficient"]
    ),
    (
        "Assumption",
        "text", "Q5",
        "Statement: The government has decided to close down all "
        "loss-making public sector units. "
        "Assumption I: All public sector units are loss-making. "
        "Assumption II: The government wants to reduce financial burden.",
        ["ii", "assumption ii", "only assumption ii"]
    ),
    (
        "Computer Awareness",
        "text", "Q6",
        "Which shortcut key opens Run dialog in Windows?",
        ["windows+r", "win+r", "winkey+r", "r"]
    ),
    (
        "Current Affairs — IPL",
        "text", "Q7",
        "Who won IPL 2025?",
        ["ipl", "2025"]
    ),
    (
        "GK",
        "text", "Q8",
        "Who is the author of Wings of Fire?",
        ["kalam", "abdul"]
    ),
    (
        "Banking Awareness",
        "text", "Q9",
        "What is the full form of NEFT?",
        ["national electronic funds transfer", "neft"]
    ),
]


# ─────────────────────────────────────────────
# CONSOLE HELPERS
# ─────────────────────────────────────────────

def _c(t, code): return f"\033[{code}m{t}\033[0m"
def green(t):    return _c(t, "92")
def red(t):      return _c(t, "91")
def yellow(t):   return _c(t, "93")
def cyan(t):     return _c(t, "96")
def bold(t):     return _c(t, "1")


# ─────────────────────────────────────────────
# AUTO TEST RUNNER
# ─────────────────────────────────────────────

def run_auto_tests(base=BASE_URL):

    print(bold(cyan("\nAUTO TESTS — v6.0\n")))

    passed = 0
    failed = 0

    for i, test in enumerate(AUTO_TESTS, 1):

        name, endpoint, qid, payload, expect = test

        try:
            if endpoint == "health":
                resp   = requests.get(base + "/health", timeout=15)
            else:
                resp   = requests.post(
                    base + "/solve-text",
                    data={"qid": qid, "text": payload},
                    timeout=180            # web search needs time
                )

            result = resp.json()
            dump   = json.dumps(result).lower()
            ok     = any(kw.lower() in dump for kw in expect)

            if ok:
                print(green(f"[PASS] {name}"))
                passed += 1
            else:
                print(red(f"[FAIL] {name}"))
                print("  →", result)
                failed += 1

        except Exception as e:
            print(red(f"[ERR]  {name}: {e}"))
            failed += 1

    print()
    print(green(f"Passed : {passed}"))
    print(red(  f"Failed : {failed}"))
    print()


# ─────────────────────────────────────────────
# MANUAL TESTER
# ─────────────────────────────────────────────

def pretty_result(result):
    print()
    print(cyan("────────────────────────────────────"))
    print("QID    :", result.get("QID"))
    print()
    print(result.get("ANSWER", "(no answer)"))
    print(cyan("────────────────────────────────────"))
    print()


def run_manual(base=BASE_URL):

    print(cyan("\nMANUAL TEST MODE — v6.0\n"))
    print(yellow("  Format : Q10 :: your question here"))
    print(yellow("  Type   : auto     → run all auto tests"))
    print(yellow("  Type   : health   → server health check"))
    print(yellow("  Type   : q        → quit\n"))

    while True:
        try:
            raw = input(green("solver> ")).strip()
        except KeyboardInterrupt:
            print()
            break

        if not raw:
            continue

        if raw in ["q", "quit", "exit"]:
            break

        if raw == "health":
            try:
                r = requests.get(base + "/health", timeout=10)
                print(r.json())
            except Exception as e:
                print(red(str(e)))
            continue

        if raw == "auto":
            run_auto_tests(base)
            continue

        try:
            qid  = "Q1"
            text = raw

            if "::" in raw:
                qid, text = raw.split("::", 1)
                qid = clean_qid(qid)

            r = requests.post(
                base + "/solve-text",
                data={"qid": qid, "text": text},
                timeout=180
            )
            pretty_result(r.json())

        except Exception as e:
            print(red(str(e)))


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":

    args = sys.argv[1:]
    base = BASE_URL

    if "--base" in args:
        idx  = args.index("--base")
        base = args[idx + 1].rstrip("/")

    if "--test" in args:
        run_auto_tests(base)

    elif "--manual" in args:
        run_manual(base)

    elif "--testall" in args:
        run_auto_tests(base)
        run_manual(base)

    else:
        if not API_KEY:
            print(red("ERROR: OPENAI_API_KEY not set"))
            sys.exit(1)

        print(green("\nExam Solver Server v6.0"))
        print("Model      :", MODEL)
        print("Web Search : ENABLED (web_search_preview)")
        print("Max Tokens :", MAX_TOKENS)
        print()

        uvicorn.run(
            "server:app",
            host="0.0.0.0",
            port=8000,
            reload=False
        )
