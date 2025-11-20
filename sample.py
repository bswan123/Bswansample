# sample.py
import os
import re
import json
import base64
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
import openai

app = FastAPI(title="Multi-Image MCQ Solver")

# --- OpenAI client init (project-scoped optional) ---
API_KEY = os.getenv("OPENAI_API_KEY")
PROJECT_ID = os.getenv("OPENAI_PROJECT_ID")
client = None
if API_KEY:
    client = openai.OpenAI(api_key=API_KEY, project=PROJECT_ID) if PROJECT_ID else openai.OpenAI(api_key=API_KEY)

# --- Strict system prompt (JSON only) ---
SYSTEM_PROMPT = """
You are an OCR + reasoning assistant specialized in multi-image multiple-choice questions (MCQs).
Rules (follow exactly):
1) Each request contains images that together form ONE MCQ: question text, diagram(s), and options (A,B,C,D,E).
2) ALWAYS parse text in images (OCR). Use arithmetic rules (BODMAS/PEMDAS) for calculations.
3) If images are blurry/missing text → return status \"unclear\".
4) If more than one option could be correct → return status \"confused\".
5) Return ONLY JSON, no extra text, using exactly this schema:
{
  "question_number": <integer|null>,
  "total_images": <integer>,
  "status": "ok" | "unclear" | "confused",
  "correct_option": "A" | "B" | "C" | "D" | "E" | null,
  "explanation": "<short one-line reason or null>"
}
6) If you determine the correct option, set status=\"ok\", correct_option to the letter, explanation = short reason (max 20 words). If unclear/confused set correct_option=null.
7) Do NOT include any other fields or non-JSON text.
"""

@app.get("/", response_class=JSONResponse)
def root():
    return {"message": "Multi-image MCQ solver active"}

@app.get("/test", response_class=HTMLResponse)
def test_page():
    html = """
    <html><head><meta charset="utf-8"><title>MCQ Solver Test</title></head>
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
          if (!filesEl.files.length) { alert('Choose at least one image'); return; }
          const fd = new FormData();
          if (qnum) fd.append('qnum', qnum);
          for (let i=0;i<filesEl.files.length;i++){ fd.append('files', filesEl.files[i]); }
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
    </body></html>
    """
    return HTMLResponse(content=html)

def extract_question_number_from_filename(filename: Optional[str]):
    """
    Robust filename -> question number extractor.
    Rules:
    1) If filename contains pattern like q5 or Q5 -> return 5
    2) Else find first integer in filename but only accept it if <= 999 (to avoid timestamps).
    3) Otherwise return None.
    """
    if not filename:
        return None

    # try q<num> pattern first (q5, Q12)
    m = re.search(r'[qQ][\-_ ]?(\d{1,3})', filename)
    if m:
        try:
            return int(m.group(1))
        except:
            pass

    # find any integer substrings and accept small reasonable numbers
    all_nums = re.findall(r'(\d+)', filename)
    for num in all_nums:
        try:
            n = int(num)
        except:
            continue
        # ignore huge numbers (timestamps etc). Accept reasonable question numbers up to 999
        if 1 <= n <= 999:
            return n

    return None

def try_parse_json_candidate(text: str):
    """Try to extract a JSON object from text; return dict or None."""
    # First try raw parse
    try:
        j = json.loads(text)
        if isinstance(j, dict):
            return j
    except Exception:
        pass
    # If the model wrapped JSON in text, search for first { ... } substring:
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            j = json.loads(candidate)
            if isinstance(j, dict):
                return j
        except Exception:
            pass
    return None

def sanitize_and_build_response(parsed: dict, qnum: Optional[int], total_images: int):
    # default shape
    out = {
        "question_number": qnum,
        "total_images": total_images,
        "status": "confused",
        "correct_option": None,
        "explanation": None
    }
    if not parsed:
        return out
    # copy fields if valid
    status = parsed.get("status")
    if status in ("ok","unclear","confused"):
        out["status"] = status
    if parsed.get("correct_option") and isinstance(parsed.get("correct_option"), str):
        opt = parsed.get("correct_option").strip().upper()
        if opt in ("A","B","C","D","E"):
            out["correct_option"] = opt
    if parsed.get("explanation"):
        out["explanation"] = str(parsed.get("explanation"))[:200]
    return out

def fallback_extract_letter(text: str):
    m = re.search(r'\b([A-E])\b', text, re.IGNORECASE)
    return m.group(1).upper() if m else None

@app.post("/solve")
async def solve(files: List[UploadFile] = File(...), qnum: Optional[str] = Form(None)):
    # Determine question number: qnum form field -> first filename -> None
    q_number = None
    if qnum:
        try:
            q_number = int(qnum)
        except:
            q_number = None

    if files:
        detected = extract_question_number_from_filename(files[0].filename)
        if not q_number and detected:
            q_number = detected

    total_images = len(files) if files else 0

    # If OpenAI client missing -> helpful error
    if client is None:
        return JSONResponse({
            "question_number": q_number,
            "total_images": total_images,
            "status": "unclear",
            "correct_option": None,
            "explanation": "OPENAI_API_KEY not set in env"
        }, status_code=500)

    # Convert files to data URLs for image input
    imgs = []
    for f in files:
        data = await f.read()
        b64 = base64.b64encode(data).decode()
        imgs.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    # Build user message instructing strict JSON output
    user_text = (
        "These images together form a single MCQ (question + options). "
        "Read all images carefully, perform calculations if needed, and respond with EXACT JSON matching the schema in the system prompt. "
        "The first uploaded filename may contain the question number; if present, use it. "
        "Respond only with the required JSON."
    )

    try:
        res = client.chat.completions.create(
            model="gpt-5",  # use gpt-5 per request; if unavailable replace with gpt-4o
            messages=[
                {"role":"system", "content": SYSTEM_PROMPT},
                {"role":"user", "content": [ {"type":"text","text": user_text} ] + imgs}
            ],
        )

        # Extract model text
        raw = ""
        try:
            raw = res.choices[0].message.content.strip()
        except Exception:
            raw = str(res)

        # Try robust JSON parsing
        parsed = try_parse_json_candidate(raw)
        if parsed:
            out = sanitize_and_build_response(parsed, q_number, total_images)
            # If status ok but no correct_option extracted, try fallback letter
            if out["status"] == "ok" and out["correct_option"] is None:
                fletter = fallback_extract_letter(raw)
                if fletter:
                    out["correct_option"] = fletter
            return JSONResponse(out)

        # If no JSON parsed, try to extract a single letter as fallback
        letter = fallback_extract_letter(raw)
        if letter:
            out = {
                "question_number": q_number,
                "total_images": total_images,
                "status": "ok",
                "correct_option": letter,
                "explanation": None
            }
            return JSONResponse(out)

        # final fallback: unclear
        return JSONResponse({
            "question_number": q_number,
            "total_images": total_images,
            "status": "unclear",
            "correct_option": None,
            "explanation": raw[:200]  # small hint
        }, status_code=200)

    except Exception as e:
        # On exception, try to salvage letter from error text
        letter = fallback_extract_letter(str(e))
        if letter:
            return JSONResponse({
                "question_number": q_number,
                "total_images": total_images,
                "status": "ok",
                "correct_option": letter,
                "explanation": "extracted from exception"
            }, status_code=200)

        return JSONResponse({
            "question_number": q_number,
            "total_images": total_images,
            "status": "unclear",
            "correct_option": None,
            "explanation": str(e)[:200]
        }, status_code=500)
