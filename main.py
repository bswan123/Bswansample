#!/usr/bin/env python3

import os
import re
import sys
import base64

from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

from openai import OpenAI

import uvicorn


# =========================================================
# CONFIG
# =========================================================

API_KEY = os.environ.get("OPENAI_API_KEY", "")

MODEL = "gpt-5.4-mini"


MAX_TOKENS = 2000

LOG_DIR = "logs"

FAIL_LOG = os.path.join(LOG_DIR, "failures.txt")

os.makedirs(LOG_DIR, exist_ok=True)


app = FastAPI(
    title="Exam Solver API v7.1",
    version="7.1"
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

    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)

    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def extract_output_text(resp) -> str:

    try:

        text = resp.output_text

        if text:
            return text.strip()

    except:
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

            f.write(raw_input[:4000] + "\n")

            f.write("-" * 80 + "\n")

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
# MAIN PROMPT
# =========================================================

UNIVERSAL_PROMPT = """
You are a highly accurate exam question solver.

The input is OCR text from a question image. OCR may contain mistakes.
Correct obvious OCR errors intelligently.

Supported question types:
- arithmetic and mathematical word problems
- puzzles and seating arrangements
- English parajumble
- English error detection

CORE RULES
1. Solve accurately.
2. Return the answer only, without the QID.
3. Do not return option letters unless the question itself requires an option-letter answer.
4. Do not return JSON.
5. Do not return markdown.
6. Do not reveal chain-of-thought or hidden reasoning.
7. Do not give unnecessary explanations.
8. If a puzzle or arrangement requires a complete arrangement to establish the answer, provide the complete useful arrangement in short, TTS-friendly sentences.
9. For arithmetic, give the final numerical answer and only the minimum equation/unit needed for clarity.
10. For parajumble, give the correct sequence clearly.
11. For error detection, identify the incorrect part and the correction briefly.
12. Ignore obvious OCR garbage when the intended text is clear.

PUZZLES / SEATING ARRANGEMENTS
For every puzzle, seating arrangement, floor arrangement,
month/date arrangement, row arrangement, circular arrangement,
or other multi-variable arrangement:

Solve the entire puzzle first.

Return the COMPLETE FINAL ARRANGEMENT, even if the question asks
only one specific question about the arrangement.

Include all relevant variables such as:
person, position, facing direction, colour, city, floor,
flat, date, month, etc. whenever they are part of the puzzle.

Do not return only the answer to the final question.
For every puzzle or arrangement question:

First solve the complete puzzle.

Then return the complete final arrangement.

Do this even if the question asks only one specific question.

Include all relevant variables:
person, position, facing, colour, city, etc.

Never return only the final requested person's answer.

TTS FORMAT
Write the final result so it sounds natural when spoken through an earpiece.
Use short sentences.
Do not include the question number because the system adds the QID separately.
"""


# =========================================================
# SOLVER
# =========================================================

def solve_text_internal(
    qid: str,
    raw: str
):

    tools = []

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

        reasoning={
            "effort": "medium"
        }
    )

    ans = extract_output_text(resp)

    ans = strip_think_blocks(ans)

    ans = ans.strip()

    print()
    print("=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(ans)
    print("=" * 60)
    print()

    if looks_bad_output(ans):

        log_failure("TEXT", raw, ans)

    return {

        "qid": qid,

        "answer": ans
    }


# =========================================================
# TEXT SOLVER
# =========================================================

def call_gpt_text(qid: str, raw: str):

    raw = clean_text(raw)

    return solve_text_internal(

        qid=qid,

        raw=raw
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

        max_output_tokens=250,

        reasoning={
            "effort": "low"
        }
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

        "version": "7.1",

        "classifier": False,

        "web_search": False
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
# ENTRY
# =========================================================

if __name__ == "__main__":

    if not API_KEY:

        print("ERROR: OPENAI_API_KEY not set")

        sys.exit(1)

    print("\nExam Solver Server v7.1")
    print("Model      :", MODEL)
    print("Max Tokens :", MAX_TOKENS)
    print()

    uvicorn.run(

        "server:app",

        host="0.0.0.0",

        port=8000,

        reload=False
    )
