from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
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
      <h2>🧠 GPT Multi-Image Question Solver (Render Test)</h2>
      <form action="/solve" enctype="multipart/form-data" method="post">
        <p>Select all related images for one question:</p>
        <input name="files" type="file" multiple accept="image/*" onchange="updatePreview(event)">
        <div id="count" class="count">No images selected</div>
        <div id="preview" class="preview"></div>
        <br>
        <input type="submit" value="Upload & Solve" style="padding: 8px 16px;">
      </form>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.post("/solve")
async def solve(files: List[UploadFile] = File(...)):
    if not files:
        return JSONResponse({"status": "unclear", "correct_option": None, "explanation": "No images received."})

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
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": "These images together form one MCQ (question + options). Read all carefully and answer with the correct option (A/B/C/D/E)."}
                ] + imgs}
            ]
        )

        raw = res.choices[0].message.content.strip()
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"status": "confused", "correct_option": None, "explanation": raw}
        return parsed

    except Exception as e:
        return {"status": "unclear", "correct_option": None, "explanation": f"Error: {str(e)}"}
