from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import os, re, base64
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
# 🔹 QID CLEAN (LOOSE)
# -----------------------------
def clean_qid(qid):
    try:
        qid = str(qid)
        m = re.search(r'\d+', qid)
        return "Q" + m.group() if m else qid.strip()
    except:
        return "Q0"


# -----------------------------
# 🔹 TEXT CLEAN
# -----------------------------
def clean_text(text):
    text = text.replace(";;", " ## ")
    text = text.replace("|", " ")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# -----------------------------
# 🔹 OCR NUMBER FIX
# -----------------------------
def fix_ocr_numbers(text):

    # 1 4 → 1.4
    text = re.sub(r'(\d)\s+(\d)', r'\1.\2', text)

    # 3 1/5 → 3.2
    text = re.sub(r'(\d)\s+1/5', r'\1.2', text)

    # ratio fix
    if "ratio" in text.lower():
        text = re.sub(r'(\d)\.(\d)', r'\1:\2', text)

    return text


# -----------------------------
# 🔹 TYPE DETECTION (FIXED 🔥)
# -----------------------------
def detect_type(text):

    t = text.lower()

    # 🔥 CONTEXT FIRST (IMPORTANT)
    if any(x in t for x in ["invest", "profit", "salary", "share"]):
        return "PARTNERSHIP"

    if "ratio" in t:
        return "RATIO"

    if any(x in t for x in ["train", "speed", "distance", "km"]):
        return "TIME_WORK"

    if "%" in t or "percent" in t:
        return "PERCENTAGE"

    # 🔥 SERIES (STRICT)
    numbers = re.findall(r'\d+', t)

    if len(numbers) >= 4:
        if re.match(r'^[\d,\s;:\-]+$', t.strip()):
            return "SERIES"

    return "ARITHMETIC"


# -----------------------------
# 🔹 ANTI-SERIES OVERRIDE
# -----------------------------
def force_override_type(dtype, text):

    t = text.lower()

    if dtype == "SERIES":
        if any(word in t for word in ["invest", "salary", "profit", "ratio", "km"]):
            return "ARITHMETIC"

    return dtype


# -----------------------------
# 🔹 GARBAGE CHECK
# -----------------------------
def is_garbage(text):
    if len(text) < 5:
        return True
    if sum(c.isdigit() for c in text) == 0 and len(text) < 15:
        return True
    return False


# -----------------------------
# 🔹 PROMPT (BALANCED 🔥)
# -----------------------------
def build_prompt(qid, text, dtype):

    return f"""
You are an SSC-level math solver.

Input may have OCR errors:
- 1 4 → 1.4
- 3.4 → 3:4 (if ratio)
- missing words

Understand using context.

Rules:
- Do NOT assume blindly
- salary ≠ partner
- solve even if slightly messy
- options are optional (support only)

Detected type: {dtype}

Output:

QID: {qid}
ANS: <answer>
TYPE: {dtype}
CONF: <0-1>

No explanation.
"""


# -----------------------------
# 🔹 GPT TEXT
# -----------------------------
def solve_text_gpt(prompt):
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return res.choices[0].message.content
    except:
        return "ERROR"


# -----------------------------
# 🔹 GPT IMAGE
# -----------------------------
def solve_image_gpt(qid, image_bytes):
    try:
        b64 = base64.b64encode(image_bytes).decode()

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"QID: {qid} Solve"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                        }
                    ],
                }
            ],
        )
        return res.choices[0].message.content
    except:
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
    except:
        return {"QID": qid, "ANS": "ERROR", "TYPE": "UNKNOWN", "CONF": 0.0}


# -----------------------------
# 🔹 POST VALIDATION
# -----------------------------
def post_validate(result, text):

    t = text.lower()

    # salary safety
    if "salary" in t and "partner" not in t:
        result["CONF"] = max(result["CONF"] - 0.15, 0.1)

    if result["ANS"] == "UNKNOWN":
        result["CONF"] = min(result["CONF"], 0.3)

    return result


# -----------------------------
# 🔹 MAIN SOLVER
# -----------------------------
def solve(qid, content, is_image=False):

    qid = clean_qid(qid)

    if is_image:
        raw = solve_image_gpt(qid, content)

    else:
        content = clean_text(content)

        if is_garbage(content):
            return {
                "QID": qid,
                "ANS": "UNKNOWN",
                "TYPE": "GARBAGE",
                "CONF": 0.1
            }

        content = fix_ocr_numbers(content)

        dtype = detect_type(content)
        dtype = force_override_type(dtype, content)

        prompt = build_prompt(qid, content, dtype)

        raw = solve_text_gpt(prompt)

    if raw == "ERROR":
        return {"QID": qid, "ANS": "ERROR", "TYPE": "UNKNOWN", "CONF": 0.0}

    parsed = parse_output(raw, qid)

    parsed = post_validate(parsed, content if not is_image else "")

    return parsed


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
    return {"status": "running 🚀"}
