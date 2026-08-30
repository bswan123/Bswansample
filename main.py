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

API_KEY = os.environ.get(
    "OPENAI_API_KEY",
    ""
)

MODEL = "gpt-5.4-mini"

# IMPORTANT:
# This budget includes reasoning + visible output.
# 4000 was too small for some reasoning-heavy puzzles.
MAX_TOKENS = 12000

# OCR gets its own larger budget.
OCR_MAX_TOKENS = 4000

# Retry count when OpenAI returns no usable text.
MAX_RETRIES = 3

LOG_DIR = "logs"

FAIL_LOG = os.path.join(
    LOG_DIR,
    "failures.txt"
)

os.makedirs(
    LOG_DIR,
    exist_ok=True
)


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Exam Solver API v7.2",
    version="7.2"
)

client = OpenAI(
    api_key=API_KEY
)


# =========================================================
# HELPERS
# =========================================================

def clean_qid(qid: str) -> str:

    try:

        qid = str(
            qid
        ).strip()

        match = re.search(
            r"\d+",
            qid
        )

        if match:

            return (
                "Q"
                + match.group()
            )

        return "Q1"

    except Exception:

        return "Q1"


def clean_text(text: str) -> str:

    text = str(
        text
    )

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    replacements = {

        "0S": "OS",
        "HTIP": "HTTP",
        "RBl": "RBI",
        "UP1": "UPI",
        "lndia": "India",
        "ﬁ": "fi",
        "ﬂ": "fl",
    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )

    # Remove control characters,
    # but KEEP newline and tab.
    text = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]",
        "",
        text
    )

    # Collapse excessive spaces.
    text = re.sub(
        r"[ \t]{2,}",
        " ",
        text
    )

    # Collapse excessive blank lines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


def strip_think_blocks(text: str) -> str:

    text = str(
        text
    )

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    text = re.sub(
        r"```.*?```",
        "",
        text,
        flags=re.DOTALL
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# =========================================================
# RESPONSE STATUS HELPERS
# =========================================================

def get_response_status(resp):

    try:

        return getattr(
            resp,
            "status",
            None
        )

    except Exception:

        return None


def get_incomplete_reason(resp):

    try:

        incomplete_details = getattr(
            resp,
            "incomplete_details",
            None
        )

        if incomplete_details is None:

            return None

        reason = getattr(
            incomplete_details,
            "reason",
            None
        )

        if reason:

            return str(
                reason
            )

        if isinstance(
            incomplete_details,
            dict
        ):

            reason = incomplete_details.get(
                "reason"
            )

            if reason:

                return str(
                    reason
                )

    except Exception:

        pass

    return None


# =========================================================
# ROBUST OPENAI OUTPUT EXTRACTION
# =========================================================

def extract_output_text(resp) -> str:

    # -----------------------------------------------------
    # METHOD 1
    # Responses API convenience property
    # -----------------------------------------------------

    try:

        text = getattr(
            resp,
            "output_text",
            None
        )

        if text:

            text = str(
                text
            ).strip()

            if text:

                return text

    except Exception:

        pass


    # -----------------------------------------------------
    # METHOD 2
    # Responses API output structure
    # -----------------------------------------------------

    try:

        output = getattr(
            resp,
            "output",
            None
        )

        if output:

            parts = []

            for item in output:

                content = getattr(
                    item,
                    "content",
                    None
                )

                if not content:

                    continue

                for part in content:

                    part_type = getattr(
                        part,
                        "type",
                        ""
                    )

                    # Normal Responses API text.
                    if part_type == "output_text":

                        text = getattr(
                            part,
                            "text",
                            ""
                        )

                        if text:

                            parts.append(
                                str(text)
                            )

                    # Some SDK/object representations
                    # may expose text differently.
                    elif hasattr(
                        part,
                        "text"
                    ):

                        text = getattr(
                            part,
                            "text",
                            ""
                        )

                        if text:

                            parts.append(
                                str(text)
                            )

            if parts:

                result = "\n".join(
                    parts
                ).strip()

                if result:

                    return result

    except Exception:

        pass


    # -----------------------------------------------------
    # METHOD 3
    # Legacy Chat Completions compatibility
    # -----------------------------------------------------

    try:

        choices = getattr(
            resp,
            "choices",
            None
        )

        if choices:

            message = getattr(
                choices[0],
                "message",
                None
            )

            if message:

                content = getattr(
                    message,
                    "content",
                    None
                )

                if content:

                    if isinstance(
                        content,
                        str
                    ):

                        return content.strip()

                    # Some APIs return content blocks.
                    if isinstance(
                        content,
                        list
                    ):

                        parts = []

                        for block in content:

                            if isinstance(
                                block,
                                dict
                            ):

                                txt = block.get(
                                    "text",
                                    ""
                                )

                                if txt:

                                    parts.append(
                                        str(txt)
                                    )

                        if parts:

                            return "\n".join(
                                parts
                            ).strip()

    except Exception:

        pass


    return ""


# =========================================================
# RESPONSE DEBUG SUMMARY
# =========================================================

def print_response_debug(
    resp,
    label: str
):

    print()
    print("=" * 70)
    print(label)
    print("=" * 70)

    try:

        status = get_response_status(
            resp
        )

        reason = get_incomplete_reason(
            resp
        )

        print(
            "STATUS:",
            status
        )

        print(
            "INCOMPLETE REASON:",
            reason
        )

    except Exception:

        pass


    try:

        usage = getattr(
            resp,
            "usage",
            None
        )

        if usage:

            print(
                "USAGE:",
                usage
            )

    except Exception:

        pass


    try:

        output = getattr(
            resp,
            "output",
            None
        )

        if output:

            print(
                "OUTPUT ITEMS:",
                len(output)
            )

            for index, item in enumerate(
                output
            ):

                try:

                    item_type = getattr(
                        item,
                        "type",
                        ""
                    )

                    print(
                        f"ITEM {index}:",
                        item_type
                    )

                except Exception:

                    pass

    except Exception:

        pass


    print("=" * 70)
    print()


# =========================================================
# FAILURE LOGGER
# =========================================================

def log_failure(
    kind: str,
    raw_input: str,
    output: str
):

    try:

        with open(
            FAIL_LOG,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(
                "\n"
                + "=" * 80
                + "\n"
            )

            f.write(
                f"TIME: {datetime.now()}\n"
            )

            f.write(
                f"TYPE: {kind}\n"
            )

            f.write(
                "-" * 80
                + "\n"
            )

            f.write(
                str(
                    raw_input
                )[:8000]
                + "\n"
            )

            f.write(
                "-" * 80
                + "\n"
            )

            f.write(
                str(
                    output
                )[:8000]
                + "\n"
            )

            f.write(
                "=" * 80
                + "\n"
            )

    except Exception:

        pass


def looks_bad_output(
    text: str
) -> bool:

    t = str(
        text
    ).strip().lower()

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

    return any(
        pattern in t
        for pattern in bad_patterns
    )


# =========================================================
# MAIN PROMPT
# =========================================================

UNIVERSAL_PROMPT = """
You are a highly accurate exam question solver.

The input is OCR text from a question image.
OCR may contain mistakes.
Correct obvious OCR errors intelligently.

SUPPORTED QUESTION TYPES

- arithmetic and mathematical word problems
- puzzles and seating arrangements
- English parajumble
- English error detection

CORE RULES

1. Solve accurately.

2. Return the answer only, without the QID.

3. Do not return option letters unless the question itself
   requires an option-letter answer.

4. Do not return JSON.

5. Do not return markdown.

6. Do not reveal chain-of-thought or hidden reasoning.

7. Do not give unnecessary explanations.

8. If a puzzle or arrangement requires a complete arrangement
   to establish the answer, provide the complete useful
   arrangement in short, TTS-friendly sentences.

9. For arithmetic, give the final numerical answer and only
   the minimum equation or unit needed for clarity.

10. For parajumble, give the correct sequence clearly.

11. For error detection, identify the incorrect part and
    the correction briefly.

12. Ignore obvious OCR garbage when the intended text is clear.


PUZZLES / SEATING ARRANGEMENTS

For every puzzle, seating arrangement, floor arrangement,
month/date arrangement, row arrangement, circular arrangement,
square arrangement, or other multi-variable arrangement:

Solve the entire puzzle first.

Return the COMPLETE FINAL ARRANGEMENT, even if the question
asks only one specific question about the arrangement.

Include all relevant variables such as:

person
position
facing direction
colour
city
floor
flat
date
month

whenever they are part of the puzzle.

Do not return only the answer to the final question.

The complete arrangement is mandatory for multi-variable
puzzles because it may be needed to answer follow-up questions.


TTS FORMAT

Write the final result so it sounds natural when spoken
through an earpiece.

Use short sentences.

Do not include the question number because the system
adds the QID separately.

Do not include the QID in the answer.

Do not output JSON.
"""


# =========================================================
# OPENAI TEXT REQUEST
# =========================================================

def request_text_model(
    user_msg: str
):

    last_response = None

    last_error = None


    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        print()
        print(
            "=" * 70
        )

        print(
            f"TEXT MODEL ATTEMPT {attempt}/{MAX_RETRIES}"
        )

        print(
            "MODEL:",
            MODEL
        )

        print(
            "MAX TOKENS:",
            MAX_TOKENS
        )

        print(
            "REASONING:",
            "medium"
        )

        print(
            "=" * 70
        )


        try:

            resp = client.responses.create(

                model=MODEL,

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


            last_response = resp


            print_response_debug(
                resp,
                "OPENAI TEXT RESPONSE"
            )


            ans = extract_output_text(
                resp
            )

            ans = strip_think_blocks(
                ans
            )

            ans = ans.strip()


            print()
            print(
                "=" * 70
            )

            print(
                "EXTRACTED ANSWER"
            )

            print(
                "=" * 70
            )

            print(
                ans
                if ans
                else "[EMPTY]"
            )

            print(
                "=" * 70
            )


            if ans:

                return ans


            # -------------------------------------------------
            # Empty output.
            # Retry automatically.
            # -------------------------------------------------

            incomplete_reason = (
                get_incomplete_reason(
                    resp
                )
            )

            log_failure(
                "EMPTY_MODEL_OUTPUT_ATTEMPT",
                user_msg,
                (
                    "Attempt: "
                    + str(attempt)
                    + "\n"
                    + "Reason: "
                    + str(incomplete_reason)
                )
            )


            print(
                "WARNING: Model returned no visible text."
            )

            print(
                "Incomplete reason:",
                incomplete_reason
            )

            print(
                "Retrying..."
            )


        except Exception as e:

            last_error = e

            error_text = repr(
                e
            )

            print()
            print(
                "=" * 70
            )

            print(
                "OPENAI TEXT API ERROR"
            )

            print(
                "=" * 70
            )

            print(
                error_text
            )

            print(
                "=" * 70
            )


            log_failure(
                "OPENAI_TEXT_API_ERROR",
                user_msg,
                error_text
            )


            # Retry API errors too.
            if attempt < MAX_RETRIES:

                print(
                    "Retrying OpenAI request..."
                )

                continue

            raise


    # =====================================================
    # ALL RETRIES FAILED
    # =====================================================

    if last_error is not None:

        raise last_error


    if last_response is not None:

        reason = get_incomplete_reason(
            last_response
        )

        raise RuntimeError(
            "OpenAI returned no usable output text "
            f"after {MAX_RETRIES} attempts. "
            f"Incomplete reason: {reason}"
        )


    raise RuntimeError(
        "OpenAI returned no usable output."
    )


# =========================================================
# TEXT SOLVER
# =========================================================

def solve_text_internal(
    qid: str,
    raw: str
):

    user_msg = f"""
QID: {qid}

Question:
{raw}

Solve accurately.
"""


    ans = request_text_model(
        user_msg
    )


    if not ans:

        raise RuntimeError(
            "No answer text returned."
        )


    if looks_bad_output(
        ans
    ):

        log_failure(
            "TEXT",
            raw,
            ans
        )


    print()
    print(
        "=" * 70
    )

    print(
        "FINAL ANSWER"
    )

    print(
        "=" * 70
    )

    print(
        ans
    )

    print(
        "=" * 70
    )


    return {

        "qid": qid,

        "answer": ans
    }


# =========================================================
# TEXT SOLVER WRAPPER
# =========================================================

def call_gpt_text(
    qid: str,
    raw: str
):

    raw = clean_text(
        raw
    )

    return solve_text_internal(
        qid=qid,
        raw=raw
    )


# =========================================================
# IMAGE MODEL REQUEST
# =========================================================

def request_image_ocr(
    img_bytes: bytes,
    mime: str,
    qid: str
):

    b64 = base64.b64encode(
        img_bytes
    ).decode()


    last_error = None


    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        print()
        print(
            "=" * 70
        )

        print(
            f"IMAGE OCR ATTEMPT {attempt}/{MAX_RETRIES}"
        )

        print(
            "MODEL:",
            MODEL
        )

        print(
            "OCR MAX TOKENS:",
            OCR_MAX_TOKENS
        )

        print(
            "=" * 70
        )


        try:

            resp = client.responses.create(

                model=MODEL,

                input=[

                    {
                        "role": "user",

                        "content": [

                            {
                                "type": "input_text",

                                "text": """
Extract ONLY the English question text.

Ignore Hindi.

Ignore UI elements.

Ignore buttons.

Ignore timers.

Ignore advertisements.

Do not solve the question.

Preserve all important names,
numbers, options, constraints,
positions, colours, cities,
dates, months and directions.

Return only the extracted English
question text.
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

                max_output_tokens=OCR_MAX_TOKENS,

                reasoning={
                    "effort": "low"
                }
            )


            print_response_debug(
                resp,
                "OPENAI IMAGE RESPONSE"
            )


            ocr_text = extract_output_text(
                resp
            )

            ocr_text = strip_think_blocks(
                ocr_text
            )

            ocr_text = clean_text(
                ocr_text
            )


            print()
            print(
                "=" * 70
            )

            print(
                "OCR TEXT"
            )

            print(
                "=" * 70
            )

            print(
                ocr_text
                if ocr_text
                else "[EMPTY]"
            )

            print(
                "=" * 70
            )


            if ocr_text:

                return ocr_text


            log_failure(
                "IMAGE_OCR_EMPTY_ATTEMPT",
                qid,
                "Attempt "
                + str(attempt)
            )


        except Exception as e:

            last_error = e

            error_text = repr(
                e
            )

            print()
            print(
                "=" * 70
            )

            print(
                "OPENAI IMAGE API ERROR"
            )

            print(
                "=" * 70
            )

            print(
                error_text
            )

            print(
                "=" * 70
            )


            log_failure(
                "IMAGE_API_ERROR",
                qid,
                error_text
            )


            if attempt < MAX_RETRIES:

                print(
                    "Retrying image OCR..."
                )

                continue

            raise


    if last_error is not None:

        raise last_error


    raise RuntimeError(
        "Image model returned no OCR text "
        f"after {MAX_RETRIES} attempts."
    )


# =========================================================
# IMAGE SOLVER
# =========================================================

def call_gpt_image(
    qid: str,
    img_bytes: bytes,
    mime: str = "image/jpeg"
):

    ocr_text = request_image_ocr(
        img_bytes,
        mime,
        qid
    )


    if not ocr_text:

        raise RuntimeError(
            "Image OCR produced empty text."
        )


    # -----------------------------------------------------
    # Now solve OCR text normally.
    # -----------------------------------------------------

    return call_gpt_text(
        qid,
        ocr_text
    )


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {

        "status": "ok",

        "model": MODEL,

        "version": "7.2",

        "max_tokens": MAX_TOKENS,

        "ocr_max_tokens": OCR_MAX_TOKENS,

        "retries": MAX_RETRIES,

        "reasoning": "low",

        "classifier": False,

        "web_search": False
    }


# =========================================================
# SOLVE TEXT
# =========================================================

@app.post("/solve-text")
async def solve_text(

    qid: str = Form(
        default="Q1"
    ),

    text: str = Form(...)
):

    text = clean_text(
        text
    )


    if not text:

        return JSONResponse(

            content={
                "error": "empty text"
            },

            status_code=400
        )


    qid = clean_qid(
        qid
    )


    try:

        return call_gpt_text(
            qid,
            text
        )

    except Exception as e:

        error_text = repr(
            e
        )

        print()
        print(
            "=" * 70
        )

        print(
            "SOLVE-TEXT FINAL ERROR"
        )

        print(
            "=" * 70
        )

        print(
            error_text
        )

        print(
            "=" * 70
        )


        log_failure(
            "SOLVE_TEXT_FINAL_ERROR",
            text,
            error_text
        )


        # 502 is more accurate than pretending this
        # was a successful 200 answer.
        return JSONResponse(

            content={

                "qid": qid,

                "answer": "",

                "error": (
                    "Solver temporarily failed. "
                    "Check Render logs."
                )
            },

            status_code=502
        )


# =========================================================
# SOLVE IMAGE
# =========================================================

@app.post("/solve-image")
async def solve_image(

    qid: str = Form(
        default="Q1"
    ),

    image: UploadFile = File(...)
):

    try:

        img_bytes = await image.read()

    except Exception as e:

        return JSONResponse(

            content={
                "error":
                "could not read image"
            },

            status_code=400
        )


    if not img_bytes:

        return JSONResponse(

            content={
                "error":
                "empty image"
            },

            status_code=400
        )


    mime = (
        image.content_type
        or "image/jpeg"
    )


    qid = clean_qid(
        qid
    )


    try:

        return call_gpt_image(

            qid,

            img_bytes,

            mime
        )

    except Exception as e:

        error_text = repr(
            e
        )

        print()
        print(
            "=" * 70
        )

        print(
            "SOLVE-IMAGE FINAL ERROR"
        )

        print(
            "=" * 70
        )

        print(
            error_text
        )

        print(
            "=" * 70
        )


        log_failure(
            "SOLVE_IMAGE_FINAL_ERROR",
            qid,
            error_text
        )


        return JSONResponse(

            content={

                "qid": qid,

                "answer": "",

                "error": (
                    "Image solver temporarily failed. "
                    "Check Render logs."
                )
            },

            status_code=502
        )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    if not API_KEY:

        print(
            "ERROR: OPENAI_API_KEY not set"
        )

        sys.exit(1)


    print()
    print(
        "=" * 70
    )

    print(
        "Exam Solver Server v7.2"
    )

    print(
        "=" * 70
    )

    print(
        "Model          :",
        MODEL
    )

    print(
        "Max Tokens     :",
        MAX_TOKENS
    )

    print(
        "OCR Tokens     :",
        OCR_MAX_TOKENS
    )

    print(
        "Reasoning      :",
        "low"
    )

    print(
        "Max Retries    :",
        MAX_RETRIES
    )

    print(
        "=" * 70
    )

    print()


    uvicorn.run(

        "server:app",

        host="0.0.0.0",

        port=8000,

        reload=False
    )
