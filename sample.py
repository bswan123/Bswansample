from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from typing import List
import openai, base64, os, json

app = FastAPI()

# ---- OpenAI Project-based Auth ----
api_key = os.getenv("OPENAI_API_KEY")
project_id = os.getenv("OPENAI_PROJECT_ID")

if not api_key:
    raise RuntimeError("Missing OPENAI_API_KEY")

client = openai.OpenAI(api_key=api_key, project=project_id) if project_id else openai.OpenAI(api_key=api_key)

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
If image is blurry → status="unclear".
If more than one answer possible → status="confused".
"""

@app.get("/")
def home():
    return {"message": "✅ Render GPT Multi-Image Solver Running"}

@app.get("/test", response_class=HTMLResponse)
def upload_form():
    html = """
    <html><body style="font-family:sans-serif;">
    <h2>🧠 GPT Multi-Image Question Solver (Render Test)</h2>
    <form action="/solve" enctype="multipart/form-data" method="post">
      <p>Select all related images for one question:</p>
      <input name="files" type="file" multiple>
      <input type="submit" value="Upload & Solve">
    </form>
    </body></html>
    """
    return HTMLResponse(content=html)

@app.post("/solve")
async def solve(files: List[UploadFile] = File(...)):
    imgs = []
    for f in files:
        data = await f.read()
        imgs.append({"type":"image_url","image_url":f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"} )

    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":imgs}
            ]
        )
        raw = res.choices[0].message.content.strip()
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"status":"confused","correct_option":None,"explanation":raw}
        return parsed
    except Exception as e:
        return {"status":"unclear","correct_option":None,"explanation":str(e)}
