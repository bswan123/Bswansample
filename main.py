from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tempfile
import os
import re
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# 🔹 REQUEST MODEL
# -----------------------------
class TextRequest(BaseModel):
    qid: str
    text: str


# -----------------------------
# 🔹 SYSTEM PROMPT
# -----------------------------
SYSTEM_PROMPT = """
You are a highly accurate exam question solver.

The input may come from OCR and can contain:
- spelling mistakes
- missing symbols
- wrong formatting
- number distortions (e.g. 1 4 instead of 1.4, 3.4 instead of 3:4)

Your job:
- Understand the intended question using context
- Use answer options to correct ambiguity
- Fix OCR mistakes mentally before solving

STRICT RULES:
- Output ONLY in this format:

QID: <id>
ANS: <answer>
TYPE: <type>
CONF: <0 to 1>

- No explanation
- No extra text

CONF RULES:
- 0.9+ → very confident
- 0.7–0.9 → likely correct
- 0.4–0.7 → uncertain
- <0.4 → weak guess

TYPE must be one of:
ARITHMETIC, ALGEBRA, RATIO, AGE, TIME_WORK, PERCENTAGE, NUMBER, UNKNOWN

If unsure:
ANS: UNKNOWN
TYPE: UNKNOWN
CONF: 0.2
"""


# -----------------------------
# 🔹 GPT CALL
# -----------------------------
def ask_gpt(qid, text):
    try:
        prompt = f"QID: {qid}\n\nQuestion:\n{text}"

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content

    except:
        return "ERROR"


# -----------------------------
# 🔹 PARSER
# -----------------------------
def parse_output(raw, qid):
    try:
        lines = raw.strip().split("\n")

        data = {}
        for line in lines:
            if ":" in line:
                k, v = line.split(":", 1)
                data[k.strip()] = v.strip()

        try:
            conf = float(data.get("CONF", 0.5))
        except:
            conf = 0.5

        return {
            "QID": data.get("QID", qid),
            "ANS": data.get("ANS", "UNKNOWN"),
            "TYPE": data.get("TYPE", "UNKNOWN"),
            "CONF": round(conf, 2)
        }

    except:
        return {
            "QID": qid,
            "ANS": "ERROR",
            "TYPE": "UNKNOWN",
            "CONF": 0.0
        }


# -----------------------------
# 🔹 CONFIDENCE ADJUST
# -----------------------------
def adjust_confidence(result):
    ans = result["ANS"]
    conf = result["CONF"]

    if ans == "UNKNOWN":
        return 0.2

    if ans == "ERROR":
        return 0.0

    # numeric boost
    if ans.replace(".", "", 1).isdigit():
        conf += 0.05

    # long garbage penalty
    if len(ans) > 20:
        conf -= 0.2

    return round(max(0.0, min(conf, 1.0)), 2)


# -----------------------------
# 🔹 OPTION EXTRACTOR
# -----------------------------
def extract_options(text):
    lines = text.split("\n")
    options = []

    for line in lines:
        if any(x in line for x in ["A)", "B)", "C)", "D)"]):
            val = re.sub(r"[A-D\)\s]", "", line)
            options.append(val)

    return options


# -----------------------------
# 🔹 FUZZY MATCH
# -----------------------------
def is_close_match(ans, opt):
    try:
        a = float(ans)
        b = float(opt)

        if abs(a - b) / max(abs(a), 1) < 0.2:
            return True
    except:
        pass

    if ans.replace(":", ".") == opt.replace(":", "."):
        return True

    return False


# -----------------------------
# 🔹 OPTION VALIDATION
# -----------------------------
def validate_with_options(result, text):

    options = extract_options(text)

    if not options:
        return result

    ans = result["ANS"]
    conf = result["CONF"]

    # exact match
    if ans in options:
        result["CONF"] = min(conf + 0.1, 1.0)
        return result

    # fuzzy match
    for opt in options:
        if is_close_match(ans, opt):
            result["CONF"] = min(conf + 0.05, 1.0)
            return result

    # mismatch penalty
    result["CONF"] = max(conf - 0.2, 0.0)

    return result


# -----------------------------
# 🔹 FINAL SOLVER
# -----------------------------
def solve_question(qid: str, content: str):

    raw = ask_gpt(qid, content)

    if raw == "ERROR":
        return {
            "QID": qid,
            "ANS": "ERROR",
            "TYPE": "UNKNOWN",
            "CONF": 0.0
        }

    parsed = parse_output(raw, qid)

    parsed["CONF"] = adjust_confidence(parsed)

    parsed = validate_with_options(parsed, content)

    return parsed


# -----------------------------
# 🔹 IMAGE ENDPOINT
# -----------------------------
@app.post("/solve-image")
async def solve_image(qid: str = Form(...), file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name

        # future OCR
        content = "image OCR text"

        result = solve_question(qid, content)

        os.remove(temp_path)

        return JSONResponse(result)

    except:
        return JSONResponse({
            "QID": qid,
            "ANS": "ERROR",
            "TYPE": "UNKNOWN",
            "CONF": 0.0
        })


# -----------------------------
# 🔹 TEXT ENDPOINT
# -----------------------------
@app.post("/solve-text")
async def solve_text(req: TextRequest):
    try:
        result = solve_question(req.qid, req.text)
        return JSONResponse(result)

    except:
        return JSONResponse({
            "QID": req.qid,
            "ANS": "ERROR",
            "TYPE": "UNKNOWN",
            "CONF": 0.0
        })


# -----------------------------
# 🔹 MANUAL OCR TEST
# -----------------------------
@app.post("/test-ocr-text")
async def test_ocr_text(qid: str = Form(...), text: str = Form(...)):
    try:
        result = solve_question(qid, text)

        return {
            "input_text": text,
            "result": result
        }

    except:
        return {"status": "error"}


# -----------------------------
# 🔹 TEST IMAGE UPLOAD
# -----------------------------
@app.post("/test-upload")
async def test_upload(file: UploadFile = File(...)):
    try:
        content = await file.read()

        return {
            "filename": file.filename,
            "size": len(content),
            "status": "received"
        }

    except:
        return {"status": "error"}


# -----------------------------
# 🔹 HEALTH
# -----------------------------
@app.get("/")
def home():
    return {"status": "server running 🚀"}
