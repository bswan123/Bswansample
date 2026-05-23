#!/usr/bin/env python3
"""
EXAM SOLVER SERVER — v7.0
====================================================

NEW IN v7
---------
✔ Smart classifier routing
✔ Selective web search
✔ Faster responses
✔ Lower token usage
✔ Cleaner OCR
✔ Removes <think> blocks
✔ Better CA/GA detection
✔ Faster arithmetic/reasoning

FLOW
----
OCR/Text
↓
Classifier
↓
if CA/GA/Computer/Banking:
    GPT + Web Search
else:
    GPT only
↓
Compact answer

RUN
---
python server.py
python server.py --manual
python server.py --test
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


# =========================================================
# CONFIG
# =========================================================

BASE_URL = "https://bswansample-1-h2uw.onrender.com"

API_KEY = os.environ.get("OPENAI_API_KEY", "")

MODEL = "gpt-4o"

CLASSIFIER_MODEL = "gpt-4o-mini"

MAX_TOKENS = 300

LOG_DIR = "logs"

FAIL_LOG = os.path.join(LOG_DIR, "failures.txt")

os.makedirs(LOG_DIR, exist_ok=True)

WEB_SEARCH_TOOL = {
    "type": "web_search_preview"
}

WEB_CATEGORIES = {
    "current_affairs",
    "banking",
    "computer",
    "static_gk"
}

app = FastAPI(
    title="Exam Solver API v7",
    version="7.0"
)

client = OpenAI(api_key=API_KEY)


# =========================================================
# HELPERS
# =========================================================

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

        "0S": "OS",
        "HTIP": "HTTP",
        "RBl": "RBI",
        "UP1": "UPI",
        "prfit": "profit",
        "invst": "invest",
        "gih": "Gita",
        "lndia": "India",
        "ﬁ": "fi",
        "ﬂ": "fl",
    }

    for k, v in replacements.items():

        text = text.replace(k, v)

    text = re.sub(r'[\x00-\x1f\x7f]', '', text)

    text = re.sub(r' {2,}', ' ', text)

    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def strip_think_blocks(text: str) -> str:

    text = re.sub(
        r'<think>.*?</think>',
        '',
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def extract_output_text(resp) -> str:

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

    except:
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


# =========================================================
# CLASSIFIER
# =========================================================

def classify_question(text: str) -> str:

    prompt = f"""
Classify this IBPS/banking exam question into EXACTLY ONE category.

Categories:
- arithmetic
- reasoning
- puzzle
- syllogism
- coding
- computer
- banking
- current_affairs
- static_gk

Question:
{text}

Return ONLY category name.
"""

    try:

        resp = client.responses.create(

            model=CLASSIFIER_MODEL,

            input=prompt,

            max_output_tokens=10,

            temperature=0
        )

        out = extract_output_text(resp).strip().lower()

        valid = {

            "arithmetic",
            "reasoning",
            "puzzle",
            "syllogism",
            "coding",
            "computer",
            "banking",
            "current_affairs",
            "static_gk"
        }

        if out in valid:
            return out

    except Exception as e:

        print("classifier error:", e)

    return "reasoning"


# =========================================================
# MAIN PROMPT
# =========================================================

UNIVERSAL_PROMPT = """
You are an expert IBPS/SBI/RRB banking exam solver.

OCR may contain mistakes.
Auto-correct intelligently.

RULES:
1. Solve accurately
2. Keep answers SHORT
3. No explanations unless puzzle
4. No markdown
5. No JSON
6. If MCQ exists:
   return option letter + answer
7. For puzzle:
   return final arrangement only
8. For arithmetic:
   return final value only
9. Never refuse
"""


# =========================================================
# SOLVER
# =========================================================

def solve_text_internal(
    qid: str,
    raw: str,
    use_web: bool
):

    tools = [WEB_SEARCH_TOOL] if use_web else []

    user_msg = f"""
QID: {qid}

Question:
{raw}

Solve accurately.
"""

    resp = client.responses.create(

        model=MODEL,

        tools=tools,

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

    ans = strip_think_blocks(ans)

    if looks_bad_output(ans):

        log_failure("TEXT", raw, ans)

    return {

        "QID": qid,

        "ANSWER": ans,

        "RAW_TEXT": raw[:500]
    }


# =========================================================
# TEXT SOLVER
# =========================================================

def call_gpt_text(qid: str, raw: str):

    raw = clean_text(raw)

    category = classify_question(raw)

    use_web = category in WEB_CATEGORIES

    print()
    print("=" * 60)
    print("CATEGORY :", category)
    print("WEB      :", use_web)
    print("=" * 60)
    print()

    return solve_text_internal(

        qid=qid,

        raw=raw,

        use_web=use_web
    )


# =========================================================
# IMAGE SOLVER
# =========================================================

def call_gpt_image(
    qid: str,
    img_bytes: bytes,
    mime: str = "image/jpeg"
):

    b64 = base64.b64encode(img_bytes).decode()

    # OCR STEP

    resp = client.responses.create(

        model=MODEL,

        input=[

            {
                "role": "user",

                "content": [

                    {
                        "type": "input_text",

                        "text": """
Extract ONLY English question text.
Ignore Hindi.
Ignore UI/buttons/timers.
Do not solve.
No explanations.
"""
                    },

                    {
                        "type": "input_image",

                        "image_url":
                        f"data:{mime};base64,{b64}"
                    }
                ]
            }
        ],

        max_output_tokens=220,

        temperature=0
    )

    ocr_text = extract_output_text(resp)

    ocr_text = strip_think_blocks(ocr_text)

    print()
    print("=" * 60)
    print("OCR TEXT")
    print("=" * 60)
    print(ocr_text)
    print("=" * 60)
    print()

    return call_gpt_text(qid, ocr_text)


# =========================================================
# ROUTES
# =========================================================

@app.get("/health")
def health():

    return {

        "status": "ok",

        "model": MODEL,

        "version": "7.0",

        "classifier": True,

        "web_search": True
    }


@app.post("/solve-text")
async def solve_text(

    qid: str = Form(default="Q1"),

    text: str = Form(...)
):

    text = clean_text(text)

    if not text:

        return JSONResponse(

            {"error": "empty text"},

            status_code=400
        )

    qid = clean_qid(qid)

    return call_gpt_text(qid, text)


@app.post("/solve-image")
async def solve_image(

    qid: str = Form(default="Q1"),

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

    return call_gpt_image(
        qid,
        img_bytes,
        mime
    )


# =========================================================
# MANUAL TEST MODE
# =========================================================

def run_manual():

    while True:

        try:

            raw = input("\nsolver> ").strip()

        except KeyboardInterrupt:

            print()

            break

        if not raw:
            continue

        if raw in ["q", "quit", "exit"]:
            break

        try:

            qid = "Q1"

            text = raw

            if "::" in raw:

                qid, text = raw.split("::", 1)

                qid = clean_qid(qid)

            r = requests.post(

                BASE_URL + "/solve-text",

                data={
                    "qid": qid,
                    "text": text
                },

                timeout=180
            )

            print()
            print("=" * 60)

            print(r.json())

            print("=" * 60)

        except Exception as e:

            print(e)


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":

    if not API_KEY:

        print("ERROR: OPENAI_API_KEY not set")

        sys.exit(1)

    if "--manual" in sys.argv:

        run_manual()

    else:

        print("\nExam Solver Server v7")
        print("Model      :", MODEL)
        print("Classifier :", CLASSIFIER_MODEL)
        print("Max Tokens :", MAX_TOKENS)
        print()

        uvicorn.run(

            "server:app",

            host="0.0.0.0",

            port=8000,

            reload=False
        )
