# sample.py
import re, json, base64, os
from typing import List
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
import openai

app = FastAPI()

# --- Auth ---
api_key = os.getenv("OPENAI_API_KEY")
project_id = os.getenv("OPENAI_PROJECT_ID")
if not api_key:
    # don't crash in startup — return helpful 500 on calls instead
    api_key = None

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
    <head><title>Test</title></head>
    <body>
      <h3>Upload images to /solve</h3>
      <form action="/solve" enctype="multipart/form-data" method="post">
        <input name="files" type="file" multiple accept="image/*">
        <br><br>
        <input type="submit" value="Upload & Solve">
      </form>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

def extract_option_from_text(text: str):
    try:
        j = json.loads(text)
        if isinstance(j, dict) and j.get("correct_option"):
            return (j.get("correct_option") or "").strip().upper()
    except Exception:
        pass
    m = re.search(r'correct_option\"\s*:\s*\"?([A-E])\"?', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m2 = re.search(r'\b([A-E])\b', text, re.IGNORECASE)
    if m2:
        return m2.group(1).upper()
    return None

@app.post("/solve")
async def solve(files: List[UploadFile] = File(...)):
    # Quick env check
    if client is None:
        return JSONResponse({"error": "OPENAI_API_KEY not set in env"}, status_code=500)

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
                    {"type": "text", "text": "These images together form one MCQ (question + options). Answer A/B/C/D/E only."}
                ] + imgs}
            ],
            # timeout/other args can be added if needed
        )
        raw = res.choices[0].message.content.strip()
        opt = extract_option_from_text(raw)
        return JSONResponse({"correct_option": opt, "raw": raw})
    except Exception as e:
        return JSONResponse({"error": "model call failed", "detail": str(e)}, status_code=500)
