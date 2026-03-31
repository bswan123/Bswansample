#!/usr/bin/env python3
"""
EXAM SOLVER SERVER — v2.1
==========================
ENDPOINTS:
  /solve-image  → RPi 5 ONLINE mode (screen or written page image)
  /solve-text   → RPi Zero OFFLINE mode (LoRa OCR text)
  /health       → status

IMAGE MODES (solve-image):
  screen  → single question → ANS: "3400"
  written → page with N questions → ANS: {"Q1":"3360","Q2":"105",...}

TEXT MODES (solve-text):
  screen  → single OCR question → ANS: "3400"
  written → multiline OCR text  → ANS: {"Q1":"3360","Q2":"105",...}

RUN:
  python server.py              → start server
  python server.py --test       → auto tests
  python server.py --manual     → interactive CLI
  python server.py --testall    → auto + manual
"""

import os, sys, json, re, base64, requests
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from openai import OpenAI
import uvicorn

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_URL = "https://bswansample-1-h2uw.onrender.com"
API_KEY  = os.environ.get("OPENAI_API_KEY", "")
MODEL    = "gpt-4o"

app = FastAPI(
    title="Exam Solver API v2.1",
    description="""
## Exam Solver — IBPS / SBI / RRB

| Endpoint | Caller | Mode | ANS type |
|----------|--------|------|----------|
| `/solve-image` mode=screen  | RPi 5 ONLINE | single question image | string |
| `/solve-image` mode=written | RPi 5 ONLINE | page with N questions | dict |
| `/solve-text`  mode=screen  | RPi Zero OFFLINE | single OCR text | string |
| `/solve-text`  mode=written | RPi Zero OFFLINE | multiline OCR text | dict |
""",
    version="2.1.0",
)

client = OpenAI(api_key=API_KEY)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def clean_qid(qid: str) -> str:
    try:
        qid = str(qid).strip()
        m = re.search(r'\d+', qid)
        return "Q" + m.group() if m else (qid or "Q?")
    except Exception:
        return "Q?"


def clean_text(text: str) -> str:
    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r' {2,}', ' ', text).strip()
    return text


def detect_type_hint(text: str) -> str:
    t = text.strip()
    if ";;" in t:
        return "SERIES"
    parts = t.split(";")
    if len(parts) == 2:
        l = re.findall(r"-?\d+\.?\d*", parts[0])
        r = re.findall(r"-?\d+\.?\d*", parts[1])
        if len(l) >= 3 and len(r) >= 3:
            return "QUADRATIC"
    return "TEXT"


def _next_qid(qid: str, offset: int) -> str:
    digits = ''.join(filter(str.isdigit, qid))
    return f"Q{int(digits) + offset}" if digits else f"Q{offset + 1}"


# ─────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────

SYSTEM_SINGLE = """You are an expert solver for Indian banking competitive exams (IBPS PO, SBI PO, RRB).

Input comes from:
- Camera photo of exam screen (Qwen2-VL OCR on RPi 5)
- LoRa radio OCR text from RPi Zero (may have bit errors / OCR glitches)
- Direct image of exam screen

YOUR JOB: Fix errors internally, solve, return clean JSON.

━━━ OCR ERRORS TO FIX INTERNALLY ━━━
- "1 5 lacs" = 1.5 lacs  (space inside number, financial context)
- "3 4" = 3:4             (space in ratio context)
- "prfit"=profit, "invst"=invest, "gih"=Gita
- "x 2" = x²             (superscript separated by OCR)
- Use context — do not apply rules blindly

━━━ QUESTION TYPES ━━━
1. PUZZLE/ARRANGEMENT  — seating, floor, box, row, circular table
2. ARITHMETIC          — percentage, profit/loss, ratio, SI/CI, TSD
3. PARTNERSHIP         — invest + profit, working partner salary
4. SERIES              — number pattern, find missing/next term
5. QUADRATIC           — "a,b,c;a,b,c" format → compare roots
6. IMAGE               — question visible directly in image

━━━ RULES ━━━
- NEVER refuse. Always attempt.
- If options present (A/B/C/D or after ;;) → match ANS to option
- Completely unrecognisable → TYPE="GARBAGE", CONF=0.05, ANS="Cannot determine"

━━━ OUTPUT — ONLY valid JSON, no markdown, no explanation ━━━
{
  "QID": "as given",
  "ANS": "final answer",
  "TYPE": "PUZZLE|ARITHMETIC|PARTNERSHIP|SERIES|QUADRATIC|IMAGE|GARBAGE|OTHER",
  "CONF": 0.85,
  "STEPS": "1-2 line working"
}"""


SYSTEM_WRITTEN = """You are an expert solver for Indian banking competitive exams (IBPS PO, SBI PO, RRB).

You will receive text extracted from a handwritten or printed exam question page.
The text contains MULTIPLE questions — typically 5, but could be more or fewer.

Each line looks like:
  Q1: 2, 4, 12, 60, 420, ?
  Q2: 20, 30, 55, ?, 310
  Q3: 17, 18, 38, ?, 472, 2365
  ... and so on

YOUR JOB:
1. Read every question line carefully
2. Identify question number from line (Q1, Q2, etc.)
3. Solve each question independently
4. Return ALL answers in one JSON

━━━ COMMON TYPES ON WRITTEN PAGES ━━━
- Number series: find missing/next term
- Arithmetic: SI/CI, ratio, percentage, profit/loss
- Quadratic equations

━━━ RULES ━━━
- Use question numbers exactly as they appear in text (Q1, Q2 etc.)
- If question number not visible, use the starting_qid provided and increment
- If a question is unreadable → ANS for that QID = "?"
- Solve each independently

━━━ OUTPUT — ONLY valid JSON, no markdown, no explanation ━━━
{
  "QID": "PAGE",
  "ANS": {
    "Q1": "3360",
    "Q2": "105",
    "Q3": "94",
    "Q4": "26",
    "Q5": "83"
  },
  "TYPE": "WRITTEN_PAGE",
  "CONF": 0.88,
  "STEPS": "Q1: pattern x1*x2=x3...; Q2: diff +10+25..."
}"""


SYSTEM_WRITTEN_IMAGE = """You are an expert solver for Indian banking competitive exams (IBPS PO, SBI PO, RRB).

You will receive a photo of a handwritten or printed exam question page.
The page contains MULTIPLE questions — typically 5, but could be more or fewer.

YOUR JOB:
1. Read every question from the page carefully
2. Identify question numbers (Q1, Q2... or 1. 2. or circled numbers)
3. Solve each question independently
4. Return ALL answers in one JSON

━━━ COMMON TYPES ON WRITTEN PAGES ━━━
- Number series: find missing/next term
- Arithmetic: SI/CI, ratio, percentage, profit/loss
- Quadratic equations

━━━ RULES ━━━
- Use question numbers visible on page
- If not visible, use starting_qid provided and increment
- If a question is unreadable → ANS for that QID = "?"

━━━ OUTPUT — ONLY valid JSON, no markdown, no explanation ━━━
{
  "QID": "PAGE",
  "ANS": {
    "Q1": "3360",
    "Q2": "105",
    "Q3": "94",
    "Q4": "26",
    "Q5": "83"
  },
  "TYPE": "WRITTEN_PAGE",
  "CONF": 0.88,
  "STEPS": "Q1: pattern...; Q2: diff..."
}"""


# ─────────────────────────────────────────────
# GPT CALLS
# ─────────────────────────────────────────────

def call_gpt_single_text(qid: str, raw: str, hint: str = "") -> dict:
    """Single question text → single answer."""
    user_msg = f"QID: {qid}\n"
    if hint and hint != "TEXT":
        user_msg += f"[Hint: {hint}]\n"
    user_msg += f"\n{raw}"

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_SINGLE},
            {"role": "user",   "content": user_msg}
        ],
        temperature=0.2,
        max_tokens=400,
        response_format={"type": "json_object"}
    )
    result = json.loads(resp.choices[0].message.content)
    result["QID"] = clean_qid(result.get("QID", qid))
    return result


def call_gpt_written_text(starting_qid: str, raw: str) -> dict:
    """
    Multiline OCR text from written page → dict of answers.

    raw looks like:
      Q1: 2, 4, 12, 60, 420, ?
      Q2: 20, 30, 55, ?, 310
      ...
    """
    user_msg = (
        f"Starting QID if not visible: {starting_qid}\n\n"
        f"OCR text from written page:\n{raw}"
    )

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_WRITTEN},
            {"role": "user",   "content": user_msg}
        ],
        temperature=0.2,
        max_tokens=600,
        response_format={"type": "json_object"}
    )
    result = json.loads(resp.choices[0].message.content)
    result["QID"] = starting_qid

    # Ensure ANS is always a dict
    if not isinstance(result.get("ANS"), dict):
        result["ANS"] = {starting_qid: str(result.get("ANS", "?"))}

    return result


def call_gpt_screen_image(qid: str, img_bytes: bytes, mime: str = "image/jpeg") -> dict:
    """Single screen question image → single answer."""
    b64 = base64.b64encode(img_bytes).decode()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_SINGLE},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}},
                {"type": "text",
                 "text": f"QID: {qid}\nPhoto of bank exam question on computer screen. Solve it. Return JSON only."}
            ]}
        ],
        temperature=0.2,
        max_tokens=500,
        response_format={"type": "json_object"}
    )
    result = json.loads(resp.choices[0].message.content)
    result["QID"] = clean_qid(result.get("QID", qid))
    return result


def call_gpt_written_image(starting_qid: str, img_bytes: bytes, mime: str = "image/jpeg") -> dict:
    """Written page photo → dict of N answers."""
    b64 = base64.b64encode(img_bytes).decode()

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_WRITTEN_IMAGE},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}},
                {"type": "text",
                 "text": (
                     f"Starting QID if not visible on page: {starting_qid}\n"
                     f"Read all questions from page, solve all, return JSON with ANS as dict."
                 )}
            ]}
        ],
        temperature=0.2,
        max_tokens=900,
        response_format={"type": "json_object"}
    )
    result = json.loads(resp.choices[0].message.content)
    result["QID"] = starting_qid

    if not isinstance(result.get("ANS"), dict):
        result["ANS"] = {starting_qid: str(result.get("ANS", "?"))}

    return result


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/health", tags=["Status"], summary="Health check")
def health():
    return {"status": "ok", "model": MODEL, "version": "2.1"}


@app.post(
    "/solve-image",
    tags=["Solver"],
    summary="ONLINE — Image from RPi 5 (screen or written page)",
)
async def solve_image(
    qid  : str        = Form(default="Q1",     description="QID e.g. Q1 (written: starting QID)"),
    mode : str        = Form(default="screen", description="screen = single question | written = page with N questions"),
    image: UploadFile = File(...,              description="Camera photo of screen or written page"),
):
    """
    Called by `uploader.py` on RPi 5 in ONLINE mode.

    **mode=screen** → single question:
    ```json
    {"QID":"Q1", "ANS":"3400", "TYPE":"ARITHMETIC", "CONF":0.9, "STEPS":"..."}
    ```

    **mode=written** → N questions on page:
    ```json
    {"QID":"Q1", "ANS":{"Q1":"3360","Q2":"105","Q3":"94","Q4":"26","Q5":"83"}, "TYPE":"WRITTEN_PAGE", "CONF":0.88, "STEPS":"..."}
    ```
    """
    img_bytes = await image.read()
    if not img_bytes:
        return JSONResponse({"error": "empty image"}, status_code=400)

    mime = image.content_type or "image/jpeg"
    qid  = clean_qid(qid)
    mode = mode.strip().lower()

    if mode == "written":
        return call_gpt_written_image(qid, img_bytes, mime)
    else:
        return call_gpt_screen_image(qid, img_bytes, mime)


@app.post(
    "/solve-text",
    tags=["Solver"],
    summary="OFFLINE — LoRa OCR text via RPi Zero",
)
async def solve_text(
    qid : str = Form(default="Q1",     description="QID or starting QID for written mode"),
    text: str = Form(...,              description="OCR text — single question or multiline written page"),
    mode: str = Form(default="screen", description="screen = single question | written = multiline OCR"),
):
    """
    Called by RPi Zero after receiving OCR text via LoRa from RPi 5.

    **mode=screen** → single question text:
    ```json
    {"QID":"Q1", "ANS":"3400", "TYPE":"ARITHMETIC", "CONF":0.9, "STEPS":"..."}
    ```

    **mode=written** → multiline OCR from written page:
    ```
    text = "Q1: 2, 4, 12, 60, 420, ?\\nQ2: 20, 30, 55, ?, 310\\n..."
    ```
    Returns:
    ```json
    {"QID":"Q1", "ANS":{"Q1":"3360","Q2":"105",...}, "TYPE":"WRITTEN_PAGE", "CONF":0.88, "STEPS":"..."}
    ```
    """
    text = clean_text(text)
    if not text:
        return JSONResponse({"error": "empty text"}, status_code=400)

    qid  = clean_qid(qid)
    mode = mode.strip().lower()

    if mode == "written":
        return call_gpt_written_text(qid, text)
    else:
        hint   = detect_type_hint(text)
        return call_gpt_single_text(qid, text, hint)


# ─────────────────────────────────────────────
# AUTO TESTS
# ─────────────────────────────────────────────

AUTO_TESTS = [
    # (name, endpoint, qid, mode, payload, expected_keywords)

    ("Health check",
     "health", None, None, None,
     ["ok"]),

    ("Series — 2,4,12,60,420,?  (expect 3360)",
     "text", "Q1", "screen", "2,4,12,60,420,?",
     ["3360"]),

    ("Series — 20,30,55,?,310",
     "text", "Q2", "screen", "20,30,55,?,310",
     ["105", "SERIES"]),

    ("Series — 17,18,38,?,472,2365",
     "text", "Q3", "screen", "17,18,38,?,472,2365",
     ["SERIES"]),

    ("Series — 11,12,16,?,41,66",
     "text", "Q4", "screen", "11,12,16,?,41,66",
     ["26", "SERIES"]),

    ("Series — 2,5,12,27,54,?",
     "text", "Q5", "screen", "2,5,12,27,54,?",
     ["SERIES"]),

    ("Written page — 5 series questions at once",
     "text", "Q1", "written",
     "Q1: 2, 4, 12, 60, 420, ?\nQ2: 20, 30, 55, ?, 310\nQ3: 17, 18, 38, ?, 472, 2365\nQ4: 11, 12, 16, ?, 41, 66\nQ5: 2, 5, 12, 27, 54, ?",
     ["Q1", "Q2", "Q3", "Q4", "Q5", "WRITTEN_PAGE"]),

    ("Arithmetic — Partnership messy OCR",
     "text", "Q6", "screen",
     "gih invest 1 5 lacs salary 5000 prfit 1 4 lacs find each share",
     ["PARTNERSHIP"]),

    ("Arithmetic — TSD",
     "text", "Q7", "screen", "tsd s=60km/hr d=150km t=?",
     ["2.5", "2", "hours", "hr"]),

    ("Arithmetic — Simple Interest",
     "text", "Q8", "screen", "si p=5000 r=10 t=3 find interest",
     ["1500"]),

    ("Quadratic roots",
     "text", "Q9", "screen", "1,-5,6;1,-4,4",
     ["QUADRATIC", ">", "<", "="]),

    ("Puzzle — Seating arrangement",
     "text", "Q10", "screen",
     "8 persons sit around circular table A sits 3rd left of E B sits opposite D find who sits right of A",
     ["PUZZLE"]),

    ("Garbage input",
     "text", "Q11", "screen", "xzq !! 999 @@@",
     ["GARBAGE"]),
]


def _c(t, code): return f"\033[{code}m{t}\033[0m"
def green(t):  return _c(t, "92")
def red(t):    return _c(t, "91")
def yellow(t): return _c(t, "93")
def cyan(t):   return _c(t, "96")
def bold(t):   return _c(t, "1")
def dim(t):    return _c(t, "2")


def run_auto_tests(base: str = BASE_URL):
    print(bold(cyan("\n  ╔══════════════════════════════════════════╗")))
    print(bold(cyan("  ║       AUTO TEST SUITE v2.1               ║")))
    print(bold(cyan("  ╚══════════════════════════════════════════╝\n")))

    passed = failed = 0
    n = len(AUTO_TESTS)

    for i, (name, endpoint, qid, mode, payload, expect) in enumerate(AUTO_TESTS, 1):
        try:
            if endpoint == "health":
                resp = requests.get(base + "/health", timeout=10)
            else:
                resp = requests.post(
                    base + "/solve-text",
                    data={"qid": qid, "text": payload, "mode": mode},
                    timeout=60
                )

            result = resp.json()
            dump   = json.dumps(result).lower()
            ok     = any(kw.lower() in dump for kw in expect)
            label  = green("PASS") if ok else red("FAIL")

            print(f"  [{i:02}/{n}] [{label}]  {name}")
            if ok:
                ans = result.get("ANS", result.get("status", ""))
                # Written page: show all answers
                if isinstance(ans, dict):
                    for k, v in ans.items():
                        print(f"           {yellow(k)} → {green(str(v))}")
                else:
                    print(f"           ANS  : {green(str(ans)[:80])}")
                print(f"           CONF : {result.get('CONF', '—')}")
                passed += 1
            else:
                print(f"           Expected : {yellow(str(expect))}")
                print(f"           Got      : {red(json.dumps(result)[:150])}")
                failed += 1

        except requests.exceptions.ConnectionError:
            print(f"  [{i:02}/{n}] [{red('ERR')}]  {name}  ← server not running")
            failed += 1
        except Exception as e:
            print(f"  [{i:02}/{n}] [{red('ERR')}]  {name} → {e}")
            failed += 1
        print()

    print(bold(f"  Result : {green(str(passed)+' passed')}  {red(str(failed)+' failed')}  ({n} total)\n"))


# ─────────────────────────────────────────────
# MANUAL TESTER (CLI)
# ─────────────────────────────────────────────

HELP = """
  COMMANDS
  ──────────────────────────────────────────────────────
  t              →  solve single question text
  w              →  solve written page (multiline OCR text)
  i              →  solve screen image
  p              →  solve written page image
  h              →  health check
  auto           →  run auto test suite
  help           →  this menu
  q / exit       →  quit
  ──────────────────────────────────────────────────────
  SHORTCUTS
  t <text>                   →  single question, QID auto
  t <qid> :: <text>          →  single question with QID
  w <qid> :: <text>          →  written multiline, starting QID
  i <path>                   →  screen image, QID auto
  i <qid> :: <path>          →  screen image with QID
  p <qid> :: <path>          →  written page image, starting QID
  ──────────────────────────────────────────────────────
  EXAMPLES
  t Q1 :: 2,4,12,60,420,?
  t Q2 :: gih invest 1 5 lacs prfit 1 4 lacs
  t Q3 :: 1,-5,6;1,-4,4
  w Q1 :: Q1: 2,4,12,60,420,?\\nQ2: 20,30,55,?,310
  i Q5 :: /home/pi/screen_q5.jpg
  p Q1 :: /home/pi/written_page.jpg
  ──────────────────────────────────────────────────────
"""


def pretty_result(result: dict):
    conf = result.get("CONF", 0)
    try:
        cf = green if float(conf) >= 0.7 else (yellow if float(conf) >= 0.4 else red)
    except Exception:
        cf = yellow

    ans = result.get("ANS", "?")
    print()
    print(bold(cyan("  ┌─ RESULT ──────────────────────────────────────────")))
    print(f"  │  QID   : {bold(str(result.get('QID','?')))}")

    if isinstance(ans, dict):
        print(f"  │  ANS   : {green(bold('(written page — multiple answers)'))}")
        for k, v in sorted(ans.items(), key=lambda x: int(''.join(filter(str.isdigit, x[0])) or '0')):
            print(f"  │    {yellow(k)} → {green(bold(str(v)))}")
    else:
        print(f"  │  ANS   : {green(bold(str(ans)))}")

    print(f"  │  TYPE  : {yellow(str(result.get('TYPE','?')))}")
    print(f"  │  CONF  : {cf(str(conf))}")
    print(f"  │  STEPS : {dim(str(result.get('STEPS','—')))}")
    print(bold(cyan("  └────────────────────────────────────────────────────")))
    print()


def ask(p: str) -> str:
    return input(bold(yellow(f"  {p}: "))).strip()


def _send_image(base, qid, path, mode):
    if not os.path.exists(path):
        print(red(f"  File not found: {path}")); return
    try:
        with open(path, "rb") as f:
            data = f.read()
        ext  = path.rsplit(".", 1)[-1].lower()
        mime = {"jpg":"image/jpeg","jpeg":"image/jpeg",
                "png":"image/png","bmp":"image/bmp"}.get(ext, "image/jpeg")
        label = "WRITTEN PAGE" if mode == "written" else "SCREEN"
        print(yellow(f"\n  [{qid}] Sending {len(data)//1024}KB [{label}] →"))
        r = requests.post(
            base + "/solve-image",
            files={"image": (os.path.basename(path), data, mime)},
            data={"qid": qid, "mode": mode},
            timeout=90
        )
        pretty_result(r.json())
    except requests.exceptions.ConnectionError:
        print(red("  Server not running."))
    except Exception as e:
        print(red(f"  Error: {e}"))


def run_manual(base: str = BASE_URL):
    print(bold(cyan("\n  MANUAL TESTER v2.1 — type 'help'\n")))
    print(cyan(HELP))

    while True:
        try:
            raw = input(bold(green("\nsolver> "))).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!"); break

        if not raw:
            continue

        tokens = raw.split(None, 1)
        cmd    = tokens[0].lower()
        rest   = tokens[1].strip() if len(tokens) > 1 else ""

        if cmd in ("q", "exit", "quit"):
            print("  Bye!"); break

        elif cmd == "help":
            print(cyan(HELP))

        elif cmd == "h":
            try:
                r = requests.get(base + "/health", timeout=5)
                print(f"  {green('OK')} → {r.json()}")
            except Exception as e:
                print(red(f"  Not reachable: {e}"))

        elif cmd == "auto":
            run_auto_tests(base)

        # ── Single question text ──────────────────────────────────
        elif cmd == "t":
            if "::" in rest:
                p = rest.split("::", 1)
                qid, text = clean_qid(p[0].strip()), p[1].strip()
            elif rest:
                qid, text = "auto", rest
            else:
                qid  = clean_qid(ask("QID (Enter=auto)") or "auto")
                text = ask("Question text")
            if not text:
                print(red("  Empty.")); continue
            print(yellow(f"\n  [{qid}] [{dim('screen')}] → {text[:70]}"))
            try:
                r = requests.post(base + "/solve-text",
                                  data={"qid": qid, "text": text, "mode": "screen"},
                                  timeout=45)
                pretty_result(r.json())
            except requests.exceptions.ConnectionError:
                print(red("  Server not running."))
            except Exception as e:
                print(red(f"  Error: {e}"))

        # ── Written page multiline text ───────────────────────────
        elif cmd == "w":
            if "::" in rest:
                p = rest.split("::", 1)
                qid, text = clean_qid(p[0].strip()), p[1].strip()
            elif rest:
                qid, text = "Q1", rest
            else:
                qid  = clean_qid(ask("Starting QID e.g. Q1") or "Q1")
                print(bold("  Paste multiline OCR text. Empty line to submit:"))
                lines = []
                while True:
                    line = input()
                    if line == "":
                        break
                    lines.append(line)
                text = "\n".join(lines)
            if not text:
                print(red("  Empty.")); continue
            print(yellow(f"\n  [{qid}] [{dim('written')}] → {len(text.splitlines())} lines"))
            try:
                r = requests.post(base + "/solve-text",
                                  data={"qid": qid, "text": text, "mode": "written"},
                                  timeout=60)
                pretty_result(r.json())
            except requests.exceptions.ConnectionError:
                print(red("  Server not running."))
            except Exception as e:
                print(red(f"  Error: {e}"))

        # ── Screen image ──────────────────────────────────────────
        elif cmd == "i":
            if "::" in rest:
                p = rest.split("::", 1)
                qid, path = clean_qid(p[0].strip()), p[1].strip()
            elif rest:
                qid, path = "auto", rest
            else:
                qid  = clean_qid(ask("QID") or "auto")
                path = ask("Image path")
            _send_image(base, qid, path, "screen")

        # ── Written page image ────────────────────────────────────
        elif cmd == "p":
            if "::" in rest:
                p = rest.split("::", 1)
                qid, path = clean_qid(p[0].strip()), p[1].strip()
            elif rest:
                qid, path = "Q1", rest
            else:
                qid  = clean_qid(ask("Starting QID e.g. Q1") or "Q1")
                path = ask("Written page image path")
            _send_image(base, qid, path, "written")

        else:
            print(yellow(f"  Unknown: '{cmd}' — type 'help'"))


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
            print(red("  OPENAI_API_KEY not set!"))
            print(yellow("  export OPENAI_API_KEY='sk-...'"))
            sys.exit(1)
        print(bold(green("\n  Exam Solver Server v2.1")))
        print(f"  Docs    : {BASE_URL}/docs")
        print(f"  Model   : {MODEL}")
        print(f"  Test    : python server.py --test")
        print(f"  Manual  : python server.py --manual\n")
        uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
