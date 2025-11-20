# server_sync.py
"""
Minimal synchronous FastAPI server for multi-image MCQ solving.
- POST /upload : upload images, calls OpenAI immediately, returns JSON answer.
- GET  /test   : small HTML upload form for manual testing.

Run:
  uvicorn server_sync:app --host 0.0.0.0 --port $PORT --workers 1

Env vars:
  OPENAI_API_KEY      - required for real GPT responses
  OPENAI_PROJECT_ID   - optional (use when key is sk-proj-...)
"""

import os
import json
import base64
import traceback
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse

# Prefer modern OpenAI client, fallback to legacy openai package
try:
    from openai import OpenAI as OpenAIClient
    MODERN_OPENAI = True
except Exception:
    OpenAIClient = None
    MODERN_OPENAI = False

try:
    import openai as legacy_openai
    LEGACY_OPENAI = True
except Exception:
    legacy_openai = None
    LEGACY_OPENAI = False

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID")

# System prompt (edit if you want)
SYSTEM_PROMPT = """
You are an OCR + reasoning assistant for multi-image MCQ questions.
Each image may contain part of the question (text, diagram, or options).
Combine all images to find the correct option (A/B/C/D/E).

Return only JSON like:
{
  "status": "ok" | "unclear" | "confused",
  "correct_option": "A" | "B" | "C" | "D" | "E" | null,
  "explanation": "short reason (optional)"
}
"""

app = FastAPI(title="Sync MCQ Solver")

def make_client():
    """Return a client object (modern or legacy) or None if no API key."""
    if not OPENAI_API_KEY:
        return None
    if MODERN_OPENAI:
        if OPENAI_PROJECT_ID:
            return OpenAIClient(api_key=OPENAI_API_KEY, project=OPENAI_PROJECT_ID)
        return OpenAIClient(api_key=OPENAI_API_KEY)
    if LEGACY_OPENAI:
        legacy_openai.api_key = OPENAI_API_KEY
        return legacy_openai
    return None

@app.get("/")
def home():
    return {"message":"Sync MCQ Solver running"}

@app.get("/test", response_class=HTMLResponse)
def upload_form():
    html = """
    <html>
    <head><title>Upload Test</title></head>
    <body>
      <h3>Multi-image MCQ solver (sync)</h3>
      <form action="/upload" enctype="multipart/form-data" method="post">
        <input name="files" type="file" multiple accept="image/*"/><br/><br/>
        <input type="submit" value="Upload & Solve" />
      </form>
      <p>Uploads are processed immediately (synchronous).</p>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.post("/upload")
async def upload_and_solve(files: List[UploadFile] = File(...), batch_id: Optional[str] = Form(None), question_number: Optional[str] = Form(None)):
    # Basic validation
    if not files or len(files) == 0:
        return JSONResponse({"status":"unclear","correct_option":None,"explanation":"No images received."})

    # Build image payloads (data URLs)
    imgs = []
    for f in files:
        try:
            data = await f.read()
            b64 = base64.b64encode(data).decode()
            imgs.append({"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}})
        except Exception as e:
            return JSONResponse({"status":"unclear","correct_option":None,"explanation":f"Failed to read image: {e}"})

    client = make_client()
    if not client:
        # No API key configured -> return simulated result so you can test pipeline
        return JSONResponse({"status":"ok","correct_option":"A","explanation":"Simulated (OPENAI_API_KEY not configured)."})

    try:
        # Build messages: system + user (user includes image payload)
        system_msg = {"role":"system","content":SYSTEM_PROMPT}
        user_content = [{"type":"text","text":"These images together form one MCQ (question + options). Read and answer with the correct option (A/B/C/D/E)."}] + imgs

        if MODERN_OPENAI:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",   # change to a model you have access to
                messages=[system_msg, {"role":"user","content": user_content}],
                max_tokens=300,
                temperature=0
            )
            # try to read text content
            try:
                raw = resp.choices[0].message.content.strip()
            except Exception:
                raw = str(resp)
        else:
            # legacy API
            resp = client.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":"These images together form one MCQ. Answer with JSON."}],
                max_tokens=300,
                temperature=0
            )
            raw = resp["choices"][0]["message"]["content"].strip()

        # Try parse JSON
        try:
            parsed = json.loads(raw)
            return JSONResponse(parsed)
        except Exception:
            # fall
