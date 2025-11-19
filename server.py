# server.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from typing import List, Optional
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
    return {"message": "✅ GPT Multi-Image Solver API (Production) Active"}

@app.get("/test", response_class=HTMLResponse)
def upload_form():
    html = """
    <html><head><title>GPT Multi-Image Solver Test</title></head><body>
    <h2>Upload images</h2>
    <form action="/solve" enctype="multipart/form-data" method="post">
      <input name="files" type="file" multiple accept="image/*">
      <br><br>
      Question number (optional): <input name="question_number" type="text">
      <br><br>
      <input type="submit" value="Upload & Solve">
    </form>
    </body></html>
    """
    return HTMLResponse(content=html)

@app.post("/solve")
async def solve(files: List[UploadFile] = File(...), question_number: Optional[str] = Form(None)):
    if not files:
        return JSONResponse({"status": "unclear", "correct_option": None, "explanation": "No images received."})

    imgs = []
    for f in files:
        data = await f.read()
        # use base64 data URI so OpenAI can accept inline images
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
                    {"type": "text", "text": "These images together form one MCQ (question + options). Read all carefully and answer with the correct option (A/B/C/D/E)."}
                ] + imgs}
            ],
            max_tokens=512,
        )

        raw = res.choices[0].message.content.strip()
        try:
            parsed = json.loads(raw)
        except Exception:
            # fallback if model responded in prose: mark confused and return text
            parsed = {"status": "confused", "correct_option": None, "explanation": raw}

        # include question_number in response if provided by client
        if question_number:
            parsed["question_number"] = question_number

        # Return normalized keys
        return JSONResponse(parsed)

    except Exception as e:
        return JSONResponse({"status": "unclear", "correct_option": None, "explanation": f"Error: {str(e)}"})
