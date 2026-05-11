#!/usr/bin/env python3
"""
EXAM SOLVER SERVER — v3.0
========================================

FEATURES
--------
✔ gpt-5-mini → arithmetic/puzzle/series
✔ gpt-5 → GK/current affairs/computer/banking
✔ OCR cleanup
✔ MCQ optimization
✔ image + text solving
✔ LoRa compatible
✔ manual testing
✔ auto tests
✔ Adda247 compatible

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

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

from openai import OpenAI

import uvicorn


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

BASE_URL = "https://bswansample-1-h2uw.onrender.com"

API_KEY = os.environ.get("OPENAI_API_KEY", "")

FAST_MODEL = "gpt-5-mini"
FACT_MODEL = "gpt-5"

FACT_TYPES = {
    "CURRENT_AFFAIRS",
    "BANKING_AWARENESS",
    "COMPUTER",
    "STATIC_GK",
    "GK"
}

app = FastAPI(
    title="Exam Solver API v3.0",
    version="3.0"
)

client = OpenAI(api_key=API_KEY)


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

    # OCR cleanup
    text = text.replace("0S", "OS")
    text = text.replace("HTIP", "HTTP")
    text = text.replace("RBl", "RBI")
    text = text.replace("UP1", "UPI")

    text = text.replace("prfit", "profit")
    text = text.replace("invst", "invest")

    text = re.sub(r'[\x00-\x1f\x7f]', '', text)
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()


def choose_model(hint: str) -> str:

    if hint in FACT_TYPES:
        return FACT_MODEL

    return FAST_MODEL


def get_max_tokens(hint: str):

    if hint in FACT_TYPES:
        return 140

    if hint in [
        "PUZZLE",
        "ARITHMETIC"
    ]:
        return 400

    return 250


def detect_type_hint(text: str) -> str:

    t = text.lower().strip()

    # ─────────────────────────────
    # QUADRATIC
    # ─────────────────────────────

    if ";;" in t:
        return "QUADRATIC"

    parts = t.split(";")

    if len(parts) == 2:

        l = re.findall(r"-?\d+\.?\d*", parts[0])
        r = re.findall(r"-?\d+\.?\d*", parts[1])

        if len(l) >= 3 and len(r) >= 3:
            return "QUADRATIC"

    # ─────────────────────────────
    # SERIES
    # ─────────────────────────────

    if "?" in t:

        nums = re.findall(r'\d+', t)

        if len(nums) >= 4:
            return "SERIES"

    # ─────────────────────────────
    # PUZZLE
    # ─────────────────────────────

    puzzle_words = [
        "sits",
        "sitting",
        "floor",
        "north",
        "south",
        "circular",
        "table",
        "arrangement",
        "row",
        "box"
    ]

    if any(k in t for k in puzzle_words):
        return "PUZZLE"

    # ─────────────────────────────
    # COMPUTER
    # ─────────────────────────────

    comp_keywords = [
        "windows",
        "excel",
        "shortcut",
        "cpu",
        "ram",
        "rom",
        "browser",
        "internet",
        "software",
        "hardware",
        "operating system",
        "microsoft",
        "keyboard",
        "https",
        "http"
    ]

    if any(k in t for k in comp_keywords):
        return "COMPUTER"

    # ─────────────────────────────
    # BANKING
    # ─────────────────────────────

    bank_keywords = [
        "rbi",
        "repo",
        "slr",
        "crr",
        "neft",
        "rtgs",
        "upi",
        "bank",
        "ombudsman",
        "committee",
        "finance bank"
    ]

    if any(k in t for k in bank_keywords):
        return "BANKING_AWARENESS"

    # ─────────────────────────────
    # STATIC GK
    # ─────────────────────────────

    gk_keywords = [
        "capital",
        "headquarter",
        "author",
        "book",
        "census",
        "isro",
        "insurance company",
        "india"
    ]

    if any(k in t for k in gk_keywords):
        return "STATIC_GK"

    # ─────────────────────────────
    # ARITHMETIC
    # ─────────────────────────────

    arithmetic_words = [
        "profit",
        "loss",
        "ratio",
        "time",
        "speed",
        "distance",
        "interest",
        "work",
        "partnership"
    ]

    if any(k in t for k in arithmetic_words):
        return "ARITHMETIC"

    return "TEXT"


def extract_json(text: str):

    text = text.strip()

    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]

    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    return json.loads(text)


# ─────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────

SYSTEM_SINGLE = """
You are an expert solver for Indian banking exams.

You solve:
- Arithmetic
- Puzzle
- Seating arrangement
- Series
- Quadratic
- Computer awareness
- Banking awareness
- Current affairs
- Static GK

RULES:
- Always answer
- Never refuse
- OCR may contain mistakes
- Read MCQ options carefully

MCQ OPTIMIZATION:
- Prefer exact option text
- Prefer exact option meaning
- Avoid unnecessary explanation

OUTPUT:
Return ONLY valid JSON.

{
  "QID":"Q1",
  "ANS":"answer",
  "TYPE":"TYPE",
  "CONF":0.90,
  "STEPS":"short reasoning"
}
"""


SYSTEM_WRITTEN = """
You are an expert banking exam solver.

You receive OCR text from a handwritten or printed page containing multiple questions.

Solve ALL questions.

Return ONLY valid JSON.

{
  "QID":"PAGE",
  "ANS":{
    "Q1":"...",
    "Q2":"..."
  },
  "TYPE":"WRITTEN_PAGE",
  "CONF":0.88,
  "STEPS":"short"
}
"""


SYSTEM_WRITTEN_IMAGE = """
You are an expert banking exam solver.

You receive an IMAGE containing multiple questions.

Read all questions carefully.
Solve all questions.

Return ONLY JSON.

{
  "QID":"PAGE",
  "ANS":{
    "Q1":"...",
    "Q2":"..."
  },
  "TYPE":"WRITTEN_PAGE",
  "CONF":0.88,
  "STEPS":"short"
}
"""


# ─────────────────────────────────────────────
# GPT CALLS
# ─────────────────────────────────────────────

def call_gpt_single_text(
    qid: str,
    raw: str,
    hint: str = ""
):

    model = choose_model(hint)

    user_msg = f"""
QID: {qid}

DetectedType: {hint}

Question:
{raw}

Return JSON only.
"""

    resp = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": SYSTEM_SINGLE
            },
            {
                "role": "user",
                "content": user_msg
            }
        ],
        max_output_tokens=get_max_tokens(hint)
    )

    txt = resp.output_text.strip()

    try:

        result = extract_json(txt)

    except:

        result = {
            "QID": qid,
            "ANS": txt[:200],
            "TYPE": hint or "OTHER",
            "CONF": 0.50,
            "STEPS": ""
        }

    result["QID"] = clean_qid(
        result.get("QID", qid)
    )

    result["RAW_TEXT"] = raw[:300]

    return result


def call_gpt_written_text(
    starting_qid: str,
    raw: str
):

    resp = client.responses.create(
        model=FAST_MODEL,
        input=[
            {
                "role": "system",
                "content": SYSTEM_WRITTEN
            },
            {
                "role": "user",
                "content": f"""
Starting QID:
{starting_qid}

OCR TEXT:
{raw}

Return JSON only.
"""
            }
        ],
        max_output_tokens=700
    )

    txt = resp.output_text.strip()

    try:

        result = extract_json(txt)

    except:

        result = {
            "QID": starting_qid,
            "ANS": {
                starting_qid: txt[:200]
            },
            "TYPE": "WRITTEN_PAGE",
            "CONF": 0.40,
            "STEPS": ""
        }

    return result


def call_gpt_screen_image(
    qid: str,
    img_bytes: bytes,
    mime: str = "image/jpeg"
):

    b64 = base64.b64encode(img_bytes).decode()

    resp = client.responses.create(
        model=FACT_MODEL,
        input=[
            {
                "role": "system",
                "content": SYSTEM_SINGLE
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"""
QID: {qid}

Read screenshot carefully.
Solve question.
Return JSON only.
"""
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime};base64,{b64}"
                    }
                ]
            }
        ],
        max_output_tokens=300
    )

    txt = resp.output_text.strip()

    try:

        result = extract_json(txt)

    except:

        result = {
            "QID": qid,
            "ANS": txt[:200],
            "TYPE": "IMAGE",
            "CONF": 0.50,
            "STEPS": ""
        }

    return result


def call_gpt_written_image(
    starting_qid: str,
    img_bytes: bytes,
    mime: str = "image/jpeg"
):

    b64 = base64.b64encode(img_bytes).decode()

    resp = client.responses.create(
        model=FAST_MODEL,
        input=[
            {
                "role": "system",
                "content": SYSTEM_WRITTEN_IMAGE
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"""
Starting QID:
{starting_qid}

Solve ALL questions.
Return JSON only.
"""
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime};base64,{b64}"
                    }
                ]
            }
        ],
        max_output_tokens=900
    )

    txt = resp.output_text.strip()

    try:

        result = extract_json(txt)

    except:

        result = {
            "QID": starting_qid,
            "ANS": {
                starting_qid: txt[:300]
            },
            "TYPE": "WRITTEN_PAGE",
            "CONF": 0.40,
            "STEPS": ""
        }

    return result


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/health")
def health():

    return {
        "status": "ok",
        "fast_model": FAST_MODEL,
        "fact_model": FACT_MODEL,
        "version": "3.0"
    }


@app.post("/solve-image")
async def solve_image(

    qid: str = Form(default="Q1"),
    mode: str = Form(default="screen"),
    image: UploadFile = File(...)

):

    img_bytes = await image.read()

    if not img_bytes:
        return JSONResponse(
            {"error": "empty image"},
            status_code=400
        )

    mime = image.content_type or "image/jpeg"

    qid = clean_qid(qid)

    mode = mode.strip().lower()

    if mode == "written":
        return call_gpt_written_image(
            qid,
            img_bytes,
            mime
        )

    return call_gpt_screen_image(
        qid,
        img_bytes,
        mime
    )


@app.post("/solve-text")
async def solve_text(

    qid: str = Form(default="Q1"),
    text: str = Form(...),
    mode: str = Form(default="screen")

):

    text = clean_text(text)

    if not text:

        return JSONResponse(
            {"error": "empty text"},
            status_code=400
        )

    qid = clean_qid(qid)

    mode = mode.strip().lower()

    if mode == "written":

        return call_gpt_written_text(
            qid,
            text
        )

    hint = detect_type_hint(text)

    return call_gpt_single_text(
        qid,
        text,
        hint
    )


# ─────────────────────────────────────────────
# AUTO TESTS
# ─────────────────────────────────────────────

AUTO_TESTS = [

    (
        "Health",
        "health",
        None,
        None,
        None,
        ["ok"]
    ),

    (
        "Series",
        "text",
        "Q1",
        "screen",
        "2,4,12,60,420,?",
        ["3360"]
    ),

    (
        "Computer",
        "text",
        "Q2",
        "screen",
        "Which Windows shortcut opens run dialog?",
        ["windows", "r"]
    ),

    (
        "Banking",
        "text",
        "Q3",
        "screen",
        "Which committee formed basis of small finance banks?",
        ["nachiket"]
    ),

    (
        "GK",
        "text",
        "Q4",
        "screen",
        "Where is ISRO headquarters situated?",
        ["bangalore"]
    )
]


def _c(t, code):
    return f"\033[{code}m{t}\033[0m"


def green(t):
    return _c(t, "92")


def red(t):
    return _c(t, "91")


def yellow(t):
    return _c(t, "93")


def cyan(t):
    return _c(t, "96")


def bold(t):
    return _c(t, "1")


def run_auto_tests(base=BASE_URL):

    print(bold(cyan("\nAUTO TESTS\n")))

    passed = 0
    failed = 0

    for i, test in enumerate(AUTO_TESTS, 1):

        name, endpoint, qid, mode, payload, expect = test

        try:

            if endpoint == "health":

                resp = requests.get(
                    base + "/health",
                    timeout=10
                )

            else:

                resp = requests.post(
                    base + "/solve-text",
                    data={
                        "qid": qid,
                        "text": payload,
                        "mode": mode
                    },
                    timeout=60
                )

            result = resp.json()

            dump = json.dumps(result).lower()

            ok = any(
                kw.lower() in dump
                for kw in expect
            )

            if ok:
                print(green(f"[PASS] {name}"))
                passed += 1
            else:
                print(red(f"[FAIL] {name}"))
                print(result)
                failed += 1

        except Exception as e:

            print(red(f"[ERR] {name}: {e}"))
            failed += 1

    print()
    print(green(f"Passed: {passed}"))
    print(red(f"Failed: {failed}"))
    print()


# ─────────────────────────────────────────────
# MANUAL TESTER
# ─────────────────────────────────────────────

def pretty_result(result):

    print()

    print(cyan("────────────────────────────"))

    print("QID   :", result.get("QID"))

    print("TYPE  :", result.get("TYPE"))

    print("ANS   :", result.get("ANS"))

    print("CONF  :", result.get("CONF"))

    print("STEPS :", result.get("STEPS"))

    print(cyan("────────────────────────────"))

    print()


def run_manual(base=BASE_URL):

    print(cyan("\nMANUAL TEST MODE\n"))

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

                r = requests.get(base + "/health")

                print(r.json())

            except Exception as e:

                print(e)

            continue

        if raw == "auto":

            run_auto_tests(base)

            continue

        try:

            qid = "Q1"

            text = raw

            if "::" in raw:

                qid, text = raw.split("::", 1)

                qid = clean_qid(qid)

            r = requests.post(
                base + "/solve-text",
                data={
                    "qid": qid,
                    "text": text,
                    "mode": "screen"
                },
                timeout=60
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

        idx = args.index("--base")

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

            print(red("OPENAI_API_KEY not set"))

            sys.exit(1)

        print(green("\nExam Solver Server v3.0"))

        print("Fast model :", FAST_MODEL)

        print("Fact model :", FACT_MODEL)

        print()

        uvicorn.run(
            "server:app",
            host="0.0.0.0",
            port=8000,
            reload=False
        )
