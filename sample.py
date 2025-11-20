# sample.py
import re, json, base64, os
from typing import List
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
import openai

app = FastAPI()

# --- Auth ---
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

@app.get("/test", response_class=HTMLResponse)
def upload_form():
    # JS will send multipart/form-data via fetch and display JSON result on page
    html = """
    <html>
    <head><meta charset="utf-8"><title>Test</title></head>
    <body style="font-family:Arial,Helvetica,sans-serif;margin:2em;">
      <h3>Upload images to /solve</h3>
      <label>Question number (optional): <input id="qnum" name="qnum" type="text" placeholder="e.g. 5"></label><br><br>
      <input id="files" name="files" type="file" multiple accept="image/*"><br><br>
      <button id="send">Upload & Solve</button>
      <pre id="out" style="background:#111;color:#eee;padding:12px;border-radius:6px;margin-top:16px;display:none;"></pre>

      <script>
        const btn = document.getElementById('send');
        const out = document.getElementById('out');
        btn.addEventListener('click', async (e) => {
          e.preventDefault();
          out.style.display = 'none';
          const qnum = document.getElementById('qnum').value;
          const filesEl = document.getElementById('files');
          if (!filesEl.files.length) {
            alert('Choose at least one image file');
            return;
          }
          const fd = new FormData();
          if (qnum) fd.append('qnum', qnum);
          for (let i=0;i<filesEl.files.length;i++){
            fd.append('files', filesEl.files[i]);
          }
          out.textContent = 'Uploading...';
          out.style.display = 'block';
          try {
            const resp = await fetch('/solve', { method: 'POST', body: fd });
            const data = await resp.json();
            out.textContent = JSON.stringify(data, null, 2);
          } catch(err) {
            out.textContent = 'Request failed:\\n' + err.toString();
          }
        });
      </script>
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

from fastapi import status

@app.post("/solve")
async def solve(files: List[UploadFile] = File(...), qnum: str = Form(None)):
    # Determine question number priority: qnum form field -> filename -> None
    q_number = None
    if qnum:
        try:
            q_number = int(qnum)
        except:
            q_number = None
    if not q_number and files:
        q_number = extract_question_number_from_filename(files[0].filename)

    if client is None:
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
        fallback_opt = extract_option_from_text(str(e))
        return JSONResponse({"question_number": q_number, "correct_option": fallback_opt, "error": str(e)}, status_code=500)
