import re
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List
import openai, base64, os, json

app = FastAPI()

api_key = os.getenv("OPENAI_API_KEY")
project_id = os.getenv("OPENAI_PROJECT_ID")
if not api_key:
    raise RuntimeError("Missing OPENAI_API_KEY")
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

def extract_question_number_from_filename(filename: str):
    if not filename:
        return None
    m = re.search(r'(\d+)', filename)
    return int(m.group(1)) if m else None

def extract_option_from_text(text: str):
    # Try to load JSON if present
    try:
        j = json.loads(text)
        if isinstance(j, dict) and j.get("correct_option"):
            return (j.get("correct_option") or "").strip().upper()
    except Exception:
        pass
    # Try to find "correct_option": "X"
    m = re.search(r'correct_option\"\s*:\s*\"?([A-E])\"?', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Find standalone letter A/B/C/D/E (bounded by non-letters/digits or quotes)
    m2 = re.search(r'\b([A-E])\b', text, re.IGNORECASE)
    if m2:
        return m2.group(1).upper()
    return None

@app.post("/solve")
async def solve(files: List[UploadFile] = File(...)):
    # get question number from first filename if present
    qnum = extract_question_number_from_filename(files[0].filename) if files else None

    imgs = []
    for f in files:
        data = await f.read()
        imgs.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"
            }
        })

    try:
        res = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": "These images together form one MCQ (question + options). Read all carefully and answer with the correct option (A/B/C/D/E). Return only the JSON described in the system prompt."}
                ] + imgs}
            ]
        )

        raw = res.choices[0].message.content.strip()
        # try parse JSON first
        correct = extract_option_from_text(raw)

        # final minimal response
        return JSONResponse({"question_number": qnum, "correct_option": correct})

    except Exception as e:
        # fallback - try to salvage any letter from exception text
        fallback_opt = extract_option_from_text(str(e))
        return JSONResponse({"question_number": qnum, "correct_option": fallback_opt})
