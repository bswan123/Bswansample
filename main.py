"""
EXAM SOLVER SERVER
==================
INPUT  → image / LoRa OCR text / series / quadratic
OUTPUT → { QID, ANS, TYPE, CONF, STEPS }

RUN MODES:
  python server.py             → start FastAPI server  (then visit /docs)
  python server.py --test      → auto test suite       (server must be running)
  python server.py --manual    → interactive CLI tester
  python server.py --testall   → auto tests then manual CLI

Hosted on Render:
  https://xyz.onrender.com/docs   → Swagger manual testing UI
"""

import os, sys, json, re, base64, requests
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
import uvicorn

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_URL = "https://bswansample-1.onrender.com"
API_KEY  = os.environ.get("OPENAI_API_KEY", "")
MODEL    = "gpt-4o"

app = FastAPI(
    title="Exam Solver API",
    description="""
## Exam Solver — IBPS / SBI / RRB

Messy input (OCR / LoRa) → GPT understands → structured answer.

### Input Types
| Type | Example |
|------|---------|
| OCR Word Problem | `gih invest 1 5 lacs prfit 1 4 lacs` |
| Series | `2,6,12,20,30;;42,44,46,50` |
| Quadratic | `1,-5,6;1,-4,4` |
| TSD | `tsd s=60 d=150 t=?` |
| Image | Upload via `/solve-image` |

### Output
```json
{ "QID": "Q1", "ANS": "...", "TYPE": "...", "CONF": 0.85, "STEPS": "..." }
```
    """,
    version="1.0.0",
)

client = OpenAI(api_key=API_KEY)


# ─────────────────────────────────────────────
# PYDANTIC MODELS  (Swagger sees these as form)
# ─────────────────────────────────────────────
class TextRequest(BaseModel):
    qid : str = "auto"
    text: str

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "summary": "OCR Partnership",
                    "value": {
                        "qid": "Q1",
                        "text": "gih invest 1 5 lacs salary 5000 prfit 1 4 lacs find each share"
                    }
                },
                {
                    "summary": "Number Series",
                    "value": {
                        "qid": "Q2",
                        "text": "2,6,12,20,30;;42,44,46,50"
                    }
                },
                {
                    "summary": "Quadratic",
                    "value": {
                        "qid": "Q3",
                        "text": "1,-5,6;1,-4,4"
                    }
                },
                {
                    "summary": "TSD",
                    "value": {
                        "qid": "Q4",
                        "text": "tsd s=60km/hr d=150km t=?"
                    }
                },
                {
                    "summary": "Simple Interest",
                    "value": {
                        "qid": "Q5",
                        "text": "si p=5000 r=10 t=3 find interest"
                    }
                }
            ]
        }


class SolveResult(BaseModel):
    QID  : str
    ANS  : str
    TYPE : str
    CONF : float
    STEPS: str


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


def detect_type_hint(text: str) -> str:
    """Minimal hint only — GPT does the actual fixing."""
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


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert solver for Indian banking competitive exams (IBPS PO, SBI PO, RRB).

INPUT comes from:
- LoRa radio transmission (may have bit errors)
- OCR from exam screen photos (low resolution, broken words)
- Manual typed shorthand

YOUR JOB: Understand the messy input, fix it internally, solve it, return clean JSON.

━━━ OCR / TRANSMISSION ERRORS TO FIX INTERNALLY ━━━
- Spaces inside numbers in financial context  →  "1 5 lacs" = 1.5 lacs, "1 4" = 1.4
- Spaces inside numbers in ratio context      →  "3 4" = 3:4, "2 3 5" = 2:3:5
- Broken/garbled words → "prfit"=profit, "gih"=Gita, "invst"=invest
- Context decides: salary in partnership ≠ share of profit (may be working partner bonus)
- Do NOT blindly apply fixes — use context to decide

━━━ INPUT TYPES ━━━
1. PARTNERSHIP  — invest + profit/salary → split by capital ratio, handle working partner
2. RATIO        — a:b:c total → find individual
3. SERIES       — "n1,n2,n3;;o1,o2,o3" → find pattern, pick from right of ;;
4. QUADRATIC    — "a,b,c;a,b,c" → solve ax²+bx+c=0 both, compare roots
5. TSD          — time/speed/distance shorthand: s=60 d=150 t=?
6. SI_CI        — simple/compound interest
7. IMAGE        — question extracted visually from exam screen

━━━ BEHAVIOR ━━━
- NEVER refuse. ALWAYS attempt.
- If completely unrecognisable → TYPE="GARBAGE", CONF=0.05, ANS="Cannot determine"
- If options present (after ;; or A/B/C/D) → match ANS to closest option

━━━ OUTPUT ━━━
Return ONLY valid JSON, no markdown, no extra text:
{
  "QID": "as given or auto",
  "ANS": "final answer",
  "TYPE": "PARTNERSHIP|RATIO|SERIES|QUADRATIC|TSD|SI_CI|IMAGE|GARBAGE|OTHER",
  "CONF": 0.85,
  "STEPS": "working in 1-2 lines"
}"""


# ─────────────────────────────────────────────
# GPT CALLS
# ─────────────────────────────────────────────
def call_gpt_text(qid: str, raw: str, hint: str = "") -> dict:
    user_msg = f"QID: {qid}\n"
    if hint and hint != "TEXT":
        user_msg += f"[Input type hint: {hint}]\n"
    user_msg += f"\n{raw}"

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg}
        ],
        temperature=0.2,
        max_tokens=400,
        response_format={"type": "json_object"}
    )
    result = json.loads(resp.choices[0].message.content)
    result["QID"] = clean_qid(result.get("QID", qid))
    return result


def call_gpt_image(qid: str, img_bytes: bytes, mime: str = "image/jpeg") -> dict:
    b64 = base64.b64encode(img_bytes).decode()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}},
                {"type": "text",
                 "text": f"QID: {qid}\nSolve the exam question in this image. Return JSON only."}
            ]}
        ],
        temperature=0.2,
        max_tokens=500,
        response_format={"type": "json_object"}
    )
    result = json.loads(resp.choices[0].message.content)
    result["QID"] = clean_qid(result.get("QID", qid))
    return result


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.get(
    "/health",
    tags=["Status"],
    summary="Health check"
)
def health():
    return {"status": "ok", "model": MODEL}


@app.post(
    "/solve-text",
    response_model=SolveResult,
    tags=["Solver"],
    summary="Solve text question (OCR / LoRa / Series / Quadratic / TSD)",
)
def solve_text(req: TextRequest):
    """
    Send any messy exam question text — OCR errors, shorthand, series, quadratic.

    **Examples you can try directly in /docs:**
    - `gih invest 1 5 lacs salary 5000 prfit 1 4 lacs find each share`
    - `2,6,12,20,30;;42,44,46,50`
    - `1,-5,6;1,-4,4`
    - `tsd s=60 d=150 t=?`
    - `si p=5000 r=10 t=3`
    """
    if not req.text.strip():
        return JSONResponse({"error": "empty text"}, status_code=400)

    qid    = clean_qid(req.qid)
    hint   = detect_type_hint(req.text)
    result = call_gpt_text(qid, req.text, hint)
    return result


@app.post(
    "/solve-image",
    response_model=SolveResult,
    tags=["Solver"],
    summary="Solve image — upload exam screenshot / camera photo",
)
async def solve_image(
    qid  : str        = Form(default="auto", description="Question ID e.g. Q1"),
    image: UploadFile = File(..., description="Exam screenshot or camera photo (jpg/png)")
):
    """
    Upload an exam question image. Works with:
    - Pi Camera photo of exam screen
    - Screenshot of exam portal
    - Photo of printed question
    """
    img_bytes = await image.read()
    if not img_bytes:
        return JSONResponse({"error": "empty image"}, status_code=400)

    mime   = image.content_type or "image/jpeg"
    result = call_gpt_image(clean_qid(qid), img_bytes, mime)
    return result


# ─────────────────────────────────────────────
# AUTO TEST SUITE
# ─────────────────────────────────────────────
AUTO_TESTS = [
    ("OCR Partnership",     "Q1", "gih invest 1 5 lacs salary 5000 prfit 1 4 lacs find each share",
     ["partnership", "lacs", "PARTNERSHIP"]),
    ("Number Series",       "Q2", "2,6,12,20,30;;42,44,46,50",
     ["42"]),
    ("Quadratic roots",     "Q3", "1,-5,6;1,-4,4",
     ["QUADRATIC", ">", "<", "="]),
    ("TSD shorthand",       "Q4", "tsd s=60km/hr d=150km t=?",
     ["2.5", "2", "hours", "hr"]),
    ("Salary ratio split",  "Q5", "a b c salary ratio 3 4 5 total 36000 find b share",
     ["12000", "14400"]),
    ("Simple Interest",     "Q6", "si p=5000 r=10 t=3 find interest",
     ["1500"]),
    ("Garbage input",       "Q7", "xzq !! 999 @@@",
     ["GARBAGE"]),
    ("Health",              None, None,
     ["ok"]),
]


def _c(t, code): return f"\033[{code}m{t}\033[0m"
def green(t):  return _c(t, "92")
def red(t):    return _c(t, "91")
def yellow(t): return _c(t, "93")
def cyan(t):   return _c(t, "96")
def bold(t):   return _c(t, "1")
def dim(t):    return _c(t, "2")


def run_auto_tests(base: str = BASE_URL):
    print(bold(cyan("\n  ╔══════════════════════════════════════╗")))
    print(bold(cyan("  ║        AUTO TEST SUITE               ║")))
    print(bold(cyan("  ╚══════════════════════════════════════╝\n")))

    passed = failed = 0
    n = len(AUTO_TESTS)

    for i, (name, qid, text, expect) in enumerate(AUTO_TESTS, 1):
        url = base + ("/health" if text is None else "/solve-text")
        try:
            if text is None:
                resp = requests.get(url, timeout=10)
            else:
                resp = requests.post(url,
                                     json={"qid": qid, "text": text},
                                     timeout=30)

            result = resp.json()
            dump   = json.dumps(result).lower()
            ok     = any(kw.lower() in dump for kw in expect)
            label  = green("PASS") if ok else red("FAIL")

            print(f"  [{i:02}/{n}] [{label}]  {name}")
            if ok:
                print(f"           ANS  : {green(str(result.get('ANS', result.get('status','')))[:80])}")
                print(f"           CONF : {result.get('CONF', '—')}")
                passed += 1
            else:
                print(f"           Expected : {yellow(str(expect))}")
                print(f"           Got      : {red(json.dumps(result)[:120])}")
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
  ────────────────────────────────────────────
  t              →  solve text  (interactive: asks QID + text)
  i              →  solve image (interactive: asks QID + path)
  h              →  health check
  auto           →  run auto test suite
  help           →  this menu
  q / exit       →  quit
  ────────────────────────────────────────────
  INLINE SHORTCUTS
  t <text>             →  QID = auto
  t <qid> :: <text>    →  with QID
  i <path>             →  QID = auto
  i <qid> :: <path>    →  with QID
  ────────────────────────────────────────────
  EXAMPLES
  t Q1 :: gih invest 1 5 lacs prfit 1 4 lacs
  t 2,6,12,20,30;;42,44,46,50
  t Q3 :: 1,-5,6;1,-4,4
  t tsd s=60 d=150 t=?
  i Q5 :: /home/pi/question.jpg
"""


def pretty_result(result: dict):
    conf = result.get("CONF", 0)
    try:
        conf_f = float(conf)
        cf = green if conf_f >= 0.7 else (yellow if conf_f >= 0.4 else red)
    except Exception:
        cf = yellow

    print()
    print(bold(cyan("  ┌─ RESULT ──────────────────────────────────")))
    print(f"  │  QID   : {bold(str(result.get('QID','?')))}")
    print(f"  │  ANS   : {green(bold(str(result.get('ANS','?'))))}")
    print(f"  │  TYPE  : {yellow(str(result.get('TYPE','?')))}")
    print(f"  │  CONF  : {cf(str(conf))}")
    print(f"  │  STEPS : {dim(str(result.get('STEPS','—')))}")
    print(bold(cyan("  └────────────────────────────────────────────")))
    print()


def ask(prompt_text: str) -> str:
    return input(bold(yellow(f"  {prompt_text}: "))).strip()


def run_manual(base: str = BASE_URL):
    print(bold(cyan("\n  MANUAL TESTER (CLI) — type 'help' for commands\n")))
    print(cyan(HELP))

    while True:
        try:
            raw = input(bold(green("\nsolver> "))).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!")
            break

        if not raw:
            continue

        tokens = raw.split(None, 1)
        cmd    = tokens[0].lower()
        rest   = tokens[1].strip() if len(tokens) > 1 else ""

        if cmd in ("q", "exit", "quit"):
            print("  Bye!")
            break

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

        elif cmd == "t":
            if "::" in rest:
                p    = rest.split("::", 1)
                qid  = clean_qid(p[0].strip())
                text = p[1].strip()
            elif rest:
                qid  = "auto"
                text = rest
            else:
                qid  = clean_qid(ask("QID (Enter to skip)") or "auto")
                text = ask("Question text")

            if not text:
                print(red("  Empty — skipping.")); continue

            print(yellow(f"\n  [{qid}] → {text[:70]}{'...' if len(text)>70 else ''}"))
            try:
                r = requests.post(base + "/solve-text",
                                  json={"qid": qid, "text": text}, timeout=30)
                pretty_result(r.json())
            except requests.exceptions.ConnectionError:
                print(red("  Server not running. Terminal 1: python server.py"))
            except Exception as e:
                print(red(f"  Error: {e}"))

        elif cmd == "i":
            if "::" in rest:
                p    = rest.split("::", 1)
                qid  = clean_qid(p[0].strip())
                path = p[1].strip()
            elif rest:
                qid  = "auto"
                path = rest
            else:
                qid  = clean_qid(ask("QID (Enter to skip)") or "auto")
                path = ask("Image file path")

            if not path:
                print(red("  No path — skipping.")); continue
            if not os.path.exists(path):
                print(red(f"  File not found: {path}")); continue

            try:
                with open(path, "rb") as f:
                    img_data = f.read()
                ext  = path.rsplit(".", 1)[-1].lower()
                mime = {"jpg":"image/jpeg","jpeg":"image/jpeg",
                        "png":"image/png","bmp":"image/bmp"}.get(ext,"image/jpeg")
                print(yellow(f"\n  [{qid}] Sending {len(img_data)//1024} KB image →"))
                r = requests.post(
                    base + "/solve-image",
                    files={"image": (os.path.basename(path), img_data, mime)},
                    data={"qid": qid},
                    timeout=60
                )
                pretty_result(r.json())
            except requests.exceptions.ConnectionError:
                print(red("  Server not running. Terminal 1: python server.py"))
            except Exception as e:
                print(red(f"  Error: {e}"))

        else:
            print(yellow(f"  Unknown: '{cmd}' — type 'help'"))


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    args = sys.argv[1:]

    # Allow --base flag: python server.py --test --base https://xyz.onrender.com
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
        print(bold(green("\n  Exam Solver Server")))
        print(f"  Docs    : https://bswansample-1.onrender.com/docs")
        print(f"  Model   : {MODEL}")
        print(f"  Test    : python server.py --test")
        print(f"  Manual  : python server.py --manual\n")
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
