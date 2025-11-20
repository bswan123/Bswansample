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

# --- Morse map for A-E (server provides this in response) ---
MORSE_MAP = {
    "A": ".-",
    "B": "-...",
    "C": "-.-.",
    "D": "-..",
    "E": "."
}

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
  "morse": "<string like .- or -... or null>",
  "explanation": "<short one-line reason or null>"
}
6) If you determine the correct option, set status=\"ok\", correct_option to the letter, morse to the correct morse string, explanation = short reason (max 20 words). If unclear/confused set correct_option=null and morse=null.
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
    if not filename:
        return None
    m = re.search(r'[qQ][\-_ ]?(\d{1,3})', filename)
    if m:
        try:
            return int(m.group(1))
        except:
            pass
    all_nums = re.findall(r'(\d+)', filename)
    for num in all_nums:
        try:
            n = int(num)
        except:
            continue
        if 1 <= n <= 999:
            return n
    return None

def try_parse_json_candidate(text: str):
    try:
        j = json.loads(text)
        if isinstance(j, dict):
            return j
    except Exception:
        pass
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
        "morse": None,
        "explanation": None
    }
    if not parsed:
        return out
    status = parsed.get("status")
    if status in ("ok","unclear","confused"):
        out["status"] = status
    if parsed.get("correct_option") and isinstance(parsed.get("correct_option"), str):
        opt = parsed.get("correct_option").strip().upper()
        if opt in ("A","B","C","D","E"):
            out["correct_option"] = opt
            out["morse"] = MORSE_MAP.get(opt)  # attach morse string
    if parsed.get("morse") and isinstance(parsed.get("morse"), str) and out["morse"] is None:
        # if model itself provided morse, accept it (but still prefer canonical map)
        out["morse"] = parsed.get("morse").strip()
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

    if client is None:
        return JSONResponse({
            "question_number": q_number,
            "total_images": total_images,
            "status": "unclear",
            "correct_option": None,
            "morse": None,
            "explanation": "OPENAI_API_KEY not set in env"
        }, status_code=500)

    imgs = []
    for f in files:
        data = await f.read()
        b64 = base64.b64encode(data).decode()
        imgs.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    user_text = (
        "These images together form a single MCQ (question + options). "
        "Read all images carefully, perform calculations if needed, and respond with EXACT JSON matching the schema in the system prompt. "
        "The first uploaded filename may contain the question number; if present, use it. "
        "Respond only with the required JSON."
    )

    try:
        res = client.chat.completions.create(
            model="gpt-5",  # change to gpt-4o if you get model-not-found
            messages=[
                {"role":"system", "content": SYSTEM_PROMPT},
                {"role":"user", "content": [ {"type":"text","text": user_text} ] + imgs}
            ],
        )

        raw = ""
        try:
            raw = res.choices[0].message.content.strip()
        except Exception:
            raw = str(res)

        parsed = try_parse_json_candidate(raw)
        if parsed:
            out = sanitize_and_build_response(parsed, q_number, total_images)
            if out["status"] == "ok" and out["correct_option"] is None:
                fletter = fallback_extract_letter(raw)
                if fletter:
                    out["correct_option"] = fletter
                    out["morse"] = MORSE_MAP.get(fletter)
            # ensure morse present when correct_option available
            if out["correct_option"] and out["morse"] is None:
                out["morse"] = MORSE_MAP.get(out["correct_option"])
            return JSONResponse(out)

        # fallback: extract single letter
        letter = fallback_extract_letter(raw)
        if letter:
            out = {
                "question_number": q_number,
                "total_images": total_images,
                "status": "ok",
                "correct_option": letter,
                "morse": MORSE_MAP.get(letter),
                "explanation": None
            }
            return JSONResponse(out)

        # final fallback unclear
        return JSONResponse({
            "question_number": q_number,
            "total_images": total_images,
            "status": "unclear",
            "correct_option": None,
            "morse": None,
            "explanation": raw[:200]
        }, status_code=200)

    except Exception as e:
        letter = fallback_extract_letter(str(e))
        if letter:
            return JSONResponse({
                "question_number": q_number,
                "total_images": total_images,
                "status": "ok",
                "correct_option": letter,
                "morse": MORSE_MAP.get(letter),
                "explanation": "extracted from exception"
            }, status_code=200)

        return JSONResponse({
            "question_number": q_number,
            "total_images": total_images,
            "status": "unclear",
            "correct_option": None,
            "morse": None,
            "explanation": str(e)[:200]
        }, status_code=500)
