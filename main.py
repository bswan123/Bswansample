# ==========================================================
# Production FastAPI Server for Multi-Image MCQ Solver
# ==========================================================
from fastapi import FastAPI, File, UploadFile
from typing import List
import openai, base64, os, json, time

app = FastAPI()

# ---- OpenAI Project-based Auth ----
api_key = os.getenv("OPENAI_API_KEY")
project_id = os.getenv("OPENAI_PROJECT_ID")

if not api_key:
    raise RuntimeError("❌ Missing OPENAI_API_KEY")

# Use OpenAI client with project ID (for new API system)
client = openai.OpenAI(api_key=api_key, project=project_id) if project_id else openai.OpenAI(api_key=api_key)

# ---- System Prompt ----
SYSTEM_PROMPT = """
You are an OCR + reasoning assistant for multi-image MCQ questions.
Each image may contain part of the question (text, diagram, or options).
Combine all image content to find the correct option (A/B/C/D/E).

Return only strict JSON like:
{
  "status": "ok" | "unclear" | "confused",
  "correct_option": "A" | "B" | "C" | "D" | "E" | null,
  "explanation": "short reason (optional)"
}
If images are unreadable or incomplete → "status":"unclear".
If multiple answers seem correct → "status":"confused".
"""

@app.get("/")
def root():
    return {"message": "✅ GPT Multi-Image Solver API (Production) Active"}

@app.post("/solve")
async def solve(files: List[UploadFile] = File(...)):
    start = time.time()
    imgs = []
    for f in files:
        b = await f.read()
        imgs.append({
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{base64.b64encode(b).decode()}"
        })

    try:
        # Send to GPT model (4o/5)
        completion = client.chat.completions.create(
            model="gpt-4o",   # change to gpt-5 if project supports
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": imgs}
            ]
        )
        raw = completion.choices[0].message.content.strip()
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"status":"confused","correct_option":None,"explanation":raw}

        parsed["response_time_sec"] = round(time.time() - start, 2)
        return parsed

    except Exception as e:
        return {
            "status":"unclear",
            "correct_option":None,
            "explanation":str(e),
            "response_time_sec": round(time.time() - start, 2)
        }
