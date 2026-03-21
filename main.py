from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import os, re, base64
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# 🔹 MODELS
# -----------------------------
class TextRequest(BaseModel):
    qid: str
    text: str


# -----------------------------
# 🔹 CLEANERS
# -----------------------------
def clean_qid(qid):
    qid = str(qid).upper()
    m = re.search(r'\d+', qid)
    return "Q" + m.group() if m else "Q0"


def clean_text(text):
    text = text.replace("|", " ")
    text = text.replace(";;", " ## ")
    text = text.replace(";", " ")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# -----------------------------
# 🔹 MORSE TYPE DETECTOR
# -----------------------------
def detect_input_type(text):
    # series
    if "##" in text or "," in text:
        return "SERIES"

    # quadratic pattern: a,b,c;d,e,f
    if re.match(r"[\d\-,]+;[\d\-,]+", text):
        return "QUADRATIC"

    return "TEXT"


# -----------------------------
# 🔹 GARBAGE DETECTOR
# -----------------------------
def is_garbage(text):
    if len(text) < 5:
        return True
    if sum(c.isdigit() for c in text) == 0 and len(text) < 20:
        return True
    return False


# -----------------------------
# 🔹 PROMPT (FINAL 🔥)
# -----------------------------
SYSTEM_PROMPT = """
You are a highly accurate SSC exam solver.

INPUT TYPES:
- OCR messy text
- Image question
- Numeric series
- Arithmetic word problems

IMPORTANT RULES:

1. ALWAYS attempt solving
2. Fix OCR mistakes:
   3.4 → 3:4
   1 5 → 1.5
   3 1/5 → 3.2

3. OPTIONS:
   - May be missing or wrong
   - Solve independently first
   - Then match if possible

4. SERIES:
   - Detect pattern even with noise

5. WORD PROBLEMS:
   - Handle short forms:
     tsd = time speed distance
     s=32km/hr
     d=4km

6. IF NOT CLEAR:
   - Guess using SSC patterns

7. GARBAGE:
   - If completely unreadable:
     TYPE: GARBAGE

CONF:
0.9 high
0.7 medium
0.4 low
<0.4 guess

OUTPUT:

QID: <id>
ANS: <answer>
TYPE: <ARITHMETIC | SERIES | RATIO | AGE | TIME_WORK | PERCENTAGE | NUMBER | QUADRATIC | GARBAGE | UNKNOWN>
CONF: <0-1>

NO explanation.
"""


# -----------------------------
# 🔹 GPT TEXT
# -----------------------------
def solve_text_gpt(qid, text):
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"QID: {qid}\n{text}"}
            ],
            temperature=0.2,
        )
        return res.choices[0].message.content
    except:
        return "ERROR"


# -----------------------------
# 🔹 GPT IMAGE (VISION)
# -----------------------------
def solve_image_gpt(qid, image_bytes):
    try:
        b64 = base64.b64encode(image_bytes).decode()

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"QID: {qid}"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                        }
                    ],
                },
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

        input_type = detect_input_type(content)

        # future: quadratic offline hook
        if input_type == "QUADRATIC":
            return {
                "QID": qid,
                "ANS": "USE_LOCAL_SOLVER",
                "TYPE": "QUADRATIC",
                "CONF": 0.9
            }

        raw = solve_text_gpt(qid, content)

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
    return {"status": "running"}
