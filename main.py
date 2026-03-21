from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import base64
from openai import OpenAI
import json
from datetime import datetime

app = FastAPI(title="SSC Messy OCR + Image Solver")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if not client.api_key:
    raise ValueError("OPENAI_API_KEY missing in .env")

class TextRequest(BaseModel):
    qid: str = "loose"
    text: str

# ────────────────────────────────────────────────
#          POWERFUL SHARED SYSTEM PROMPT
# ────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert at solving SSC / competitive exam questions from extremely bad OCR text or low-quality photos.

You will get very messy, damaged, shorthand input like:
- "gih invest 1 5 lacs salary 5000 profit 1 4 lacs"
- "34,54,23;;43,56,76,23"
- "1,-4,8;2,6,14"
- "tsd s=32 d=4 t=?"
- broken words, missing spaces, wrong dots/commas

Your tasks:
1. Intelligently fix OCR mistakes:
   - 1 4 lacs → 1.4 lakhs
   - 3 4 → 3:4 (when looks like ratio)
   - ig→big, gih→big, pr→profit, slry→salary, etc.
   - ignore repeated letters, extra symbols

2. Understand context automatically:
   - invest + profit → partnership / profit sharing
   - salary different from investment → different treatment
   - numbers before ;; and after → question + options
   - a,b;c,d;e,f → likely quadratic equations to compare roots
   - tsd, s=, d=, t=? → time speed distance
   - % sign → percentage
   - train, platform, relative speed → train problems

3. ALWAYS try to give an answer — even if 50% sure.
   - If completely garbage → TYPE: "GARBAGE"

4. For images: it's probably a handwritten or printed math question / sum / word problem.

Return **ONLY** valid JSON, nothing else (no explanation outside, no ```json):
{
  "QID": "the qid user sent or loose",
  "ANS": "final answer as string — can be number, fraction, option letter, short sentence",
  "TYPE": "one of: PARTNERSHIP, RATIO, SERIES_MATCH, QUADRATIC_COMPARE, TIME_SPEED_DISTANCE, PERCENTAGE, PROFIT_LOSS, OTHER_MATH, GARBAGE, UNKNOWN",
  "CONF": 0.1 to 0.99 — your real confidence,
  "UNDERSTOOD_AS": "1-2 line what question you think it is (for debug)",
  "SOLUTION_STEPS": "very short steps how you reached answer (for debug)"
}
"""

# ────────────────────────────────────────────────
#                  TEXT ENDPOINT
# ────────────────────────────────────────────────
@app.post("/solve-text")
async def solve_text(req: TextRequest):
    try:
        text_input = req.text.strip()
        qid = req.qid.strip()

        response = client.chat.completions.create(
            model="gpt-4o",           # ← changed to 4o — mini bahut weak hai messy case mein
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"QID: {qid}\n\nRaw input (probably bad OCR):\n{text_input}"}
            ],
            max_tokens=450,
            temperature=0.3
        )

        raw_answer = response.choices[0].message.content.strip()

        # Console ke liye manual testing print
        print("\n" + "═"*80)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] TEXT | QID: {qid}")
        print("INPUT:\n" + text_input)
        print("\nGPT RAW:\n" + raw_answer)
        print("═"*80 + "\n")

        try:
            parsed = json.loads(raw_answer)
            parsed["QID"] = qid  # force correct qid
            return JSONResponse(content=parsed)
        except:
            return JSONResponse(
                content={
                    "QID": qid,
                    "ANS": "PARSE_FAIL",
                    "TYPE": "GARBAGE",
                    "CONF": 0.0,
                    "UNDERSTOOD_AS": "GPT output was not valid JSON",
                    "SOLUTION_STEPS": raw_answer[:400]
                },
                status_code=422
            )

    except Exception as e:
        raise HTTPException(500, str(e))

# ────────────────────────────────────────────────
#                 IMAGE ENDPOINT
# ────────────────────────────────────────────────
@app.post("/solve-image")
async def solve_image(qid: str = Form("loose"), file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        base64_img = base64.b64encode(image_bytes).decode("utf-8")
        mime = file.content_type or "image/jpeg"

        response = client.chat.completions.create(
            model="gpt-4o",           # vision ke liye 4o best hai
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"QID: {qid}\nThis is most likely an SSC math / reasoning question photo (possibly handwritten). Solve it."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64_img}"}}
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.25
        )

        raw_answer = response.choices[0].message.content.strip()

        # Console print for manual check
        print("\n" + "═"*80)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] IMAGE | QID: {qid}")
        print("GPT RAW:\n" + raw_answer)
        print("═"*80 + "\n")

        try:
            parsed = json.loads(raw_answer)
            parsed["QID"] = qid
            return JSONResponse(content=parsed)
        except:
            return JSONResponse(
                content={
                    "QID": qid,
                    "ANS": "PARSE_FAIL",
                    "TYPE": "GARBAGE",
                    "CONF": 0.0,
                    "UNDERSTOOD_AS": "Invalid JSON from vision model",
                    "SOLUTION_STEPS": raw_answer[:400]
                },
                status_code=422
            )

    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "time": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
