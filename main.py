from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import os, re, base64
from openai import OpenAI

app = FastAPI()

# 🔥 SAFE CLIENT INIT
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# 🔹 MODEL
# -----------------------------
class TextRequest(BaseModel):
    qid: str
    text: str


# -----------------------------
# 🔹 CLEAN QID
# -----------------------------
def clean_qid(qid):
    try:
        qid = str(qid)
        m = re.search(r'\d+', qid)
        return "Q" + m.group() if m else qid.strip()
    except:
        return "Q0"


# -----------------------------
# 🔹 CLEAN TEXT
# -----------------------------
def clean_text(text):
    text = text.replace(";;", " ## ")
    text = text.replace("|", " ")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# -----------------------------
# 🔹 OCR FIX
# -----------------------------
def fix_ocr_numbers(text):
    text = re.sub(r'(\d)\s+(\d)', r'\1.\2', text)

    if "ratio" in text.lower():
        text = re.sub(r'(\d)\.(\d)', r'\1:\2', text)

    return text


# -----------------------------
# 🔹 TYPE DETECTION (FIXED)
# -----------------------------
def detect_type(text):

    t = text.lower()

    if any(x in t for x in ["invest", "profit", "salary", "share"]):
        return "PARTNERSHIP"

    if "ratio" in t:
        return "RATIO"

    if any(x in t for x in ["train", "speed", "distance", "km"]):
        return "TIME_WORK"

    if "%" in t:
        return "PERCENTAGE"

    # STRICT SERIES
    numbers = re.findall(r'\d+', t)
    if len(numbers) >= 4 and re.match(r'^[\d,\s;:\-]+$', t.strip()):
        return "SERIES"

    return "ARITHMETIC"


# -----------------------------
# 🔹 PROMPT
# -----------------------------
def build_prompt(qid, text, dtype):
    return f"""
Solve SSC exam question.

Fix OCR errors if needed:
- 1 4 → 1.4
- 3.4 → 3:4 (if ratio)

Rules:
- salary ≠ partner
- do NOT assume blindly
- use context
- options optional

TYPE: {dtype}

QID: {qid}
{text}

Return ONLY:

QID: {qid}
ANS: <answer>
TYPE: {dtype}
CONF: <0-1>
"""


# -----------------------------
# 🔹 GPT CALL (FIXED 🔥)
# -----------------------------
def solve_text_gpt(prompt):
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            timeout=20
        )
        return res.choices[0].message.content

    except Exception as e:
        print("🔥 GPT ERROR:", str(e))
        return "ERROR"


# -----------------------------
# 🔹 IMAGE GPT
# -----------------------------
def solve_image_gpt(qid, image_bytes):
    try:
        b64 = base64.b64encode(image_bytes).decode()

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"QID: {qid} solve"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            }]
        )

        return res.choices[0].message.content

    except Exception as e:
        print("🔥 IMAGE GPT ERROR:", str(e))
        return "ERROR"


# -----------------------------
# 🔹 PARSER
# -----------------------------
def parse_output(raw, qid):
    try:
        data = {}
        for line in raw.split("\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                data[k.strip()] = v.strip()

        return {
            "QID": data.get("QID", qid),
            "ANS": data.get("ANS", "UNKNOWN"),
            "TYPE": data.get("TYPE", "UNKNOWN"),
            "CONF": float(data.get("CONF", 0.5))
        }

    except Exception as e:
        print("🔥 PARSE ERROR:", e)
        return {"QID": qid, "ANS": "ERROR", "TYPE": "UNKNOWN", "CONF": 0.0}


# -----------------------------
# 🔹 MAIN SOLVER
# -----------------------------
def solve(qid, content, is_image=False):

    qid = clean_qid(qid)

    if is_image:
        raw = solve_image_gpt(qid, content)

    else:
        content = clean_text(content)
        content = fix_ocr_numbers(content)

        dtype = detect_type(content)

        prompt = build_prompt(qid, content, dtype)

        raw = solve_text_gpt(prompt)

    if raw == "ERROR":
        return {"QID": qid, "ANS": "ERROR", "TYPE": "UNKNOWN", "CONF": 0.0}

    return parse_output(raw, qid)


# -----------------------------
# 🔹 ENDPOINTS
# -----------------------------
@app.post("/solve-image")
async def solve_image(qid: str = Form(...), file: UploadFile = File(...)):
    img = await file.read()
    return solve(qid, img, is_image=True)


@app.post("/solve-text")
async def solve_text(req: TextRequest):
    return solve(req.qid, req.text)


@app.post("/test-ocr-text")
async def test_ocr(qid: str = Form(...), text: str = Form(...)):
    return solve(qid, text)


@app.get("/")
def home():
    return {"status": "running 🔥"}
