# sample.py
import re, json, base64, os
from typing import List
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import openai

app = FastAPI()

# --- Auth (safe startup: don't crash if API key missing) ---
api_key = os.getenv("OPENAI_API_KEY")
project_id = os.getenv("OPENAI_PROJECT_ID")
client = None
if api_key:
    client = openai.OpenAI(api_key=api_key, project=project_id) if project_id else openai.OpenAI(api_key=api_key)

SYSTEM_PROMPT = """
You are an OCR + reasoning assistant for multi-image MCQ questions.
Return JSON like:
{
  "status": "ok" | "unclear" | "confused",
  "correct_option": "A" | "B" | "C" | "D" | "E" | null,
  "explanation": "short reason (optional)"
}
"""

@app.get("/")
def home():
    return {"message": "✅ GPT Multi-Image Solver API (Production) Active"}

@app.get("/test", response_class=HTMLResponse)
def upload_form():
    html = """
    <html>
    <head><meta charset="utf-8"><title>Test</title></head>
    <body style="font-family:Arial,Helvetica,sans-serif;margin:2em;">
      <h3>Upload images to /solve</h3>
      <form action="/solve" enctype="multipart/form-data" method="post">
        Question number (optional): <input name="qnum" type="text" placeholder="e.g. 5"><br><br>
        <input name="files" type="file" multiple accept="image/*"><br><br>
        <input type="submit" value="Upload & Solve" style="padding:8px 12px;">
      </form>
      <p>Or POST to <code>/solve</code> with multipart/form-data fields <code>files</code> and optional <code>qnum</code>.</p>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

def extract_question_number_from_filename(filename: str):
    if not filename:
        return None
    m = re.search(r'(\d+)', filename)
    return int(m.group(1)) if m else None

def extract_option_from_text(text: str):
    # try JSON first
    try:
        j = json.loads(text)
        if isinstance(j, dict) and j.get("correct_option"):
            return (j.get("correct_option") or "").strip().upper()
    except Exception:
        pass
    # look for correct_option pattern
    m = re.search(r'correct_option\"\s*:\s*\"?([A-E])\"?', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # fallback: first standalone A-E
    m2 = re.search(r'\b([A-E])\b', text, re.IGNORECASE)
    if m2:
        return m2.group(1).upper()
    return None

@app.post("/solve")
async def solve(files: List[UploadFile] = File(...), qnum: str = None):
    """
    Returns only:
      { "question_number": <int|null>, "correct_option": <"A"|...|null> }
    - qnum form field is optional; otherwise we'll try to extract from filename of first uploaded file.
    """
    # get question number: priority -> qnum form field -> filename number -> None
    q_number = None
    if qnum:
        try:
            q_number = int(qnum)
        except Exception:
            q_number = None

    if not q_number and files:
        q_number = extract_question_number_from_filename(files[0].filename)

    if client is None:
        # helpful error if API key missing
        return JSONResponse({"question_number": q_number, "correct_option": None, "error": "OPENAI_API_KEY not set in env"}, status_code=500)

    imgs = []
    for f in files:
        data = await f.read()
        imgs.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"}
        })

    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": "These images together form one MCQ (question + options). Answer with the correct option A/B/C/D/E only."}
                ] + imgs}
            ],
        )
        raw = res.choices[0].message.content.strip()
        opt = extract_option_from_text(raw)
        return JSONResponse({"question_number": q_number, "correct_option": opt})
    except Exception as e:
        # try salvage option from exception text
        fallback_opt = extract_option_from_text(str(e))
        return JSONResponse({"question_number": q_number, "correct_option": fallback_opt, "error": str(e)}, status_code=500)
