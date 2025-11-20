# server_sync.py
"""
Minimal synchronous FastAPI server for multi-image MCQ solving (no queue / no workers).
- POST /solve  : upload images, call OpenAI immediately, return JSON answer.
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
from typing import List

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

# Prefer modern OpenAI client
try:
    from openai import OpenAI as OpenAIClient
    MODERN_OPENAI = True
except Exception:
    OpenAIClient = None
    MODERN_OPENAI = False

# Fallback to legacy 'openai' (best-effort) — optional
try:
    import openai as legacy_openai
    LEGACY_OPENAI = True
except Exception:
    legacy_openai = None
    LEGACY_OPENAI = False

app = FastAPI(title="GPT Multi-Image Solver (sync)")

# --- OpenAI config (from env) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID")

# If no key present, we will return a simulated result
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not set — server will return simulated responses for testing.")

# Create client (modern preferred)
def make_client():
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

client = make_client()

SYSTEM_PROMPT = """
You are an OCR + reasoning assistant for multi-image MCQ questions.
Each image may contain part of the question (text, diagram, or options).
Combine all image content to find the correct option (A/B/C/D/E).

Return only JSON like:
{
  "status": "ok" | "unclear" | "confused",
  "correct_option": "A" | "B" | "C" | "D" | "E" | null,
  "explanation": "short reason (optional)"
}
If image is blurry → status=\"unclear\".
If more than one answer possible → status=\"confused\".
"""

# -------------------- ROUTES --------------------

@app.get("/")
def home():
    return {"message": "✅ GPT Multi-Image Solver API (Production) Active"}

@app.get("/test", response_class=HTMLResponse)
def upload_form():
    html = """
    <html>
    <head>
    <title>🧠 GPT Multi-Image Solver Test</title>
    <style>
      body { font-family: sans-serif; margin: 2em; }
      .preview img { height: 120px; margin: 5px; border-radius: 8px; border: 1px solid #aaa; }
      .count { color: gray; font-size: 0.9em; margin-top: 5px; }
    </style>
    <script>
      function updatePreview(event) {
          const files = event.target.files;
          const preview = document.getElementById('preview');
          const count = document.getElementById('count');
          preview.innerHTML = '';
          for (let i = 0; i < files.length; i++) {
              const img = document.createElement('img');
              img.src = URL.createObjectURL(files[i]);
              preview.appendChild(img);
          }
          count.textContent = files.length + ' image(s) selected';
      }
    </script>
    </head>
    <body>
      <h2>🧠 GPT Multi-Image Question Solver (Test)</h2>
      <form action="/solve" enctype="multipart/form-data" method="post">
        <p>Select all related images for one question:</p>
        <input name="files" type="file" multiple accept="image/*" onchange="updatePreview(event)">
        <div id="count" class="count">No images selected</div>
        <div id="preview" class="preview"></div>
        <br>
        <input type="submit" value="Upload & Solve" style="padding: 8px 16px;">
      </form>
      <p style="color:gray;">Note: This server calls OpenAI synchronously — request will wait for the model response.</p>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.post("/solve")
async def solve(files: List[UploadFile] = File(...)):
    # Basic validation
    if not files or len(files) == 0:
        return JSONResponse({"status": "unclear", "correct_option": None, "explanation": "No images received."})

    # Build image payloads as data URLs (base64). Keep them simple for multimodal client.
    imgs = []
    for f in files:
        try:
            data = await f.read()
            b64 = base64.b64encode(data).decode()
            imgs.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        except Exception as e:
            return JSONResponse({"status": "unclear", "correct_option": None, "explanation": f"Failed to read image: {e}"})

    # If no OpenAI configured -> return simulated quick response for testing
    if client is None:
        return JSONResponse({"status": "ok", "correct_option": "A", "explanation": "Simulated (OPENAI_API_KEY not configured)."})

    # Build the messages for the model
    system_msg = {"role": "system", "content": SYSTEM_PROMPT}
    user_payload = [{"type": "text", "text": "These images together form one MCQ (question + options). Read and answer with the correct option (A/B/C/D/E)."}] + imgs

    try:
        # Modern client path (OpenAIClient) — best-effort attempt to access response text
        if MODERN_OPENAI:
            resp = client.chat.completions.create(
                model="gpt-4o",  # change to model you have access to (gpt-4o-mini, gpt-4o, etc.)
                messages=[system_msg, {"role": "user", "content": user_payload}],
                max_tokens=300,
                temperature=0
            )
            # Try to read standard fields; this may vary with SDK versions — fallback to string
            try:
                raw = resp.choices[0].message.content.strip()
            except Exception:
                raw = str(resp)

        # Legacy openai package path
        else:
            resp = client.ChatCompletion.create(
                model="gpt-4o",
                messages=[system_msg, {"role": "user", "content": "These images together form one MCQ. Answer with JSON."}],
                max_tokens=300,
                temperature=0
            )
            raw = resp["choices"][0]["message"]["content"].strip()

        # Try to parse JSON output
        try:
            parsed = json.loads(raw)
            # Ensure keys exist
            return JSONResponse(parsed)
        except Exception:
            # If model didn't return JSON, return a structured fallback
            return JSONResponse({"status": "confused", "correct_option": None, "explanation": raw})

    except Exception as e:
        # Return the error in JSON; keep trace for debugging (not recommended in prod or public)
        tb = traceback.format_exc()
        return JSONResponse({"status": "unclear", "correct_option": None, "explanation": f"OpenAI error: {str(e)}", "trace": tb})
