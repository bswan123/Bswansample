# ==========================================================
# Production FastAPI Server for Multi-Image MCQ Solver (Stable Fixed Version)
# ==========================================================
from fastapi import FastAPI, File, UploadFile
from typing import List
import openai, base64, os, json, time

app = FastAPI(title="GPT Multi-Image Solver API")

# ---------- AUTH SETUP ----------
api_key = os.getenv("OPENAI_API_KEY")
project_id = os.getenv("OPENAI_PROJECT_ID")

if not api_key:
    raise RuntimeError("❌ Missing OPENAI_API_KEY environment variable.")

# Create client (new SDK supports project ID optionally)
client = openai.OpenAI(api_key=api_key, project=project_id) if project_id else openai.OpenAI(api_key=api_key)

# ---------- SYSTEM PROMPT ----------
SYSTEM_PROMPT = """
You are an OCR + reasoning assistant for multi-image MCQ questions.
Each image may contain part of the question (text, diagram, or options).
Combine all image content carefully and identify the correct option (A/B/C/D/E).

Return only strict JSON like:
{
  "status": "ok" | "unclear" | "confused",
  "correct_option": "A" | "B" | "C" | "D" | "E" | null,
  "explanation": "short reason (optional)"
}

Rules:
- If any image text is unreadable or incomplete → status = "unclear".
- If more than one answer seems correct → status = "confused".
- Do NOT include any extra commentary outside JSON.
"""

# ---------- ROUTES ----------

@app.get("/")
def root():
    """Health check"""
    return {"message": "✅ GPT Multi-Image Solver API (Production) Active"}

@app.post("/solve")
async def solve(files: List[UploadFile] = File(...)):
    """Accept multiple image files and return the solved MCQ result."""
    start_time = time.time()

    # Encode images in correct multimodal format
    imgs = []
    for f in files:
        try:
            b = await f.read()
            imgs.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64.b64encode(b).decode()}"
                }
            })
        except Exception as e:
            return {"status": "unclear", "correct_option": None, "explanation": f"File read error: {e}"}

    # Send request to OpenAI
    try:
        completion = client.chat.completions.create(
            model="gpt-5-search-api",   # or "gpt-5" if available in your org
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": "These images together form one MCQ. Read all carefully and choose the correct option (A–E)."}
                ] + imgs}
            ]
        )

        raw = completion.choices[0].message.content.strip()
        print("\n========== RAW GPT RESPONSE ==========\n", raw, "\n======================================\n")

        # Parse GPT output safely
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {
                "status": "confused",
                "correct_option": None,
                "explanation": raw
            }

        parsed["response_time_sec"] = round(time.time() - start_time, 2)
        return parsed

    except Exception as e:
        return {
            "status": "unclear",
            "correct_option": None,
            "explanation": f"Server or GPT error: {e}",
            "response_time_sec": round(time.time() - start_time, 2)
        }
