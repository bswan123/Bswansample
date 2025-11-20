# server_simple.py
"""
Minimal queue server with 2 worker threads and small task IDs.
Run:
  uvicorn server_simple:app --host 0.0.0.0 --port $PORT --workers 1

Env vars:
  OPENAI_API_KEY      - required for real GPT calls
  OPENAI_PROJECT_ID   - optional (use if your key is sk-proj-...)
  WORKER_COUNT        - optional (default 2)
  UPLOAD_ROOT         - optional (default /tmp/uploads)
"""

import os, uuid, time, json, shutil, traceback
from pathlib import Path
from threading import Thread, Lock
import queue
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse

# OpenAI client support: prefer modern client, fallback to legacy
try:
    from openai import OpenAI as OpenAIClient
    MODERN_OPENAI = True
except Exception:
    OpenAIClient = None
    MODERN_OPENAI = False

try:
    import openai as legacy_openai
    LEGACY_OPENAI = True
except Exception:
    legacy_openai = None
    LEGACY_OPENAI = False

# Configs and envs
UPLOAD_ROOT = Path(os.environ.get("UPLOAD_ROOT", "/tmp/uploads"))
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

WORKER_COUNT = int(os.environ.get("WORKER_COUNT", "2"))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_PROJECT_ID = os.environ.get("OPENAI_PROJECT_ID")

# In-memory state
TASK_QUEUE = queue.Queue()
TASKS = {}    # task_id -> status
RESULTS = {}  # task_id -> result
CANCELLED = set()
PROCESSING = set()

app = FastAPI(title="Simple 2-worker Exam Solver")

# small task id generator
_counter = 0
_counter_lock = Lock()
def next_task_id():
    global _counter
    with _counter_lock:
        _counter += 1
        return f"t{_counter:04d}"   # t0001, t0002, ...

# helpers for file save/read
def save_upload_files(task_id: str, files: List[UploadFile]):
    dest = UPLOAD_ROOT / task_id
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, f in enumerate(files):
        ext = Path(f.filename).suffix or ".jpg"
        target = dest / f"img_{i}{ext}"
        with target.open("wb") as fh:
            fh.write(f.file.read())
        paths.append(str(target))
    return paths

def write_result_file(task_id: str, payload: dict):
    p = UPLOAD_ROOT / task_id / "result.json"
    try:
        p.write_text(json.dumps(payload))
    except Exception:
        pass

def read_result_file(task_id: str):
    p = UPLOAD_ROOT / task_id / "result.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None

# build an openai client instance (call inside worker to avoid sharing issues)
def make_openai_client():
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

# worker function (no multiprocessing; OpenAI call happens in the thread)
def worker_loop(worker_idx: int):
    print(f"[worker-{worker_idx}] started")
    while True:
        task = TASK_QUEUE.get()
        task_id = task["task_id"]
        image_paths = task["image_paths"]
        qnum = task.get("question_number")
        if task_id in CANCELLED:
            TASKS[task_id] = "cancelled"
            TASK_QUEUE.task_done()
            continue

        try:
            TASKS[task_id] = "processing"
            PROCESSING.add(task_id)
            # Build prompt/messages (simple)
            system_prompt = "You are an expert exam solver. Given the images, return JSON: {\"status\":\"ok\",\"correct_option\":\"A|B|C|D|E\",\"explanation\":\"...\"}"

            # Read images as base64 dataurls (small sample; advisable to compress on Pi)
            imgs = []
            for p in image_paths:
                try:
                    b = Path(p).read_bytes()
                    import base64
                    imgs.append({"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{base64.b64encode(b).decode()}"}})
                except Exception:
                    pass

            client = make_openai_client()
            if not client:
                # simulated result (for testing without API)
                res = {"status":"ok","correct_option":"A","explanation":"simulated (no OPENAI_API_KEY)"}
                RESULTS[task_id] = res
                TASKS[task_id] = "done"
                write_result_file(task_id, {"status":"done","result":res})
                PROCESSING.discard(task_id)
                TASK_QUEUE.task_done()
                continue

            # Perform ChatCompletion call (modern or legacy)
            text = None
            try:
                if MODERN_OPENAI:
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",   # change to model you have access to
                        messages=[{"role":"system","content":system_prompt},
                                  {"role":"user","content":[{"type":"text","text":"These images together form one MCQ. Answer with JSON."}] + imgs}],
                        max_tokens=200, temperature=0
                    )
                    try:
                        text = resp.choices[0].message.content.strip()
                    except Exception:
                        text = str(resp)
                else:
                    resp = client.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=[{"role":"system","content":system_prompt},
                                  {"role":"user","content":"These images together form one MCQ. Answer with JSON."}],
                        max_tokens=200, temperature=0
                    )
                    text = resp["choices"][0]["message"]["content"].strip()
            except Exception as e:
                # log & mark failed
                TASKS[task_id] = "failed"
                RESULTS[task_id] = {"error": str(e)}
                write_result_file(task_id, {"status":"failed","error":str(e),"trace":traceback.format_exc()})
                PROCESSING.discard(task_id)
                TASK_QUEUE.task_done()
                continue

            # parse output
            try:
                parsed = json.loads(text)
            except Exception:
                import re
                m = re.search(r"[A-E]", text.upper()) if isinstance(text, str) else None
                parsed = {"status":"confused","correct_option": m.group(0) if m else None, "explanation": text}

            RESULTS[task_id] = parsed
            TASKS[task_id] = "done"
            write_result_file(task_id, {"status":"done","result":parsed})
        except Exception as e:
            TASKS[task_id] = "failed"
            RESULTS[task_id] = {"error": str(e)}
            write_result_file(task_id, {"status":"failed","error":str(e),"trace":traceback.format_exc()})
        finally:
            PROCESSING.discard(task_id)
            TASK_QUEUE.task_done()

# spawn workers at startup
@app.on_event("startup")
def startup():
    for i in range(WORKER_COUNT):
        t = Thread(target=worker_loop, args=(i,), daemon=True)
        t.start()
    print(f"Simple server started with {WORKER_COUNT} workers")

# ROUTES
@app.get("/")
def home():
    return {"message":"Simple 2-worker server active"}

@app.get("/test", response_class=HTMLResponse)
def test_form():
    return HTMLResponse("""
    <html><body>
    <h3>Upload test</h3>
    <form action="/upload" enctype="multipart/form-data" method="post">
    <input name="files" type="file" multiple accept="image/*"/>
    <input type="submit" value="Upload"/>
    </form></body></html>
    """)

@app.post("/upload")
async def upload_endpoint(files: List[UploadFile] = File(...), batch_id: Optional[str] = Form(None), question_number: Optional[str] = Form(None)):
    task_id = next_task_id()
    try:
        paths = save_upload_files(task_id, files)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed saving files: {e}")
    TASKS[task_id] = "queued"
    TASK_QUEUE.put({"task_id": task_id, "image_paths": paths, "batch_id": batch_id, "question_number": question_number})
    return JSONResponse({"task_id": task_id, "status":"queued"})

@app.get("/result/{task_id}")
def result_endpoint(task_id: str):
    status = TASKS.get(task_id)
    res = RESULTS.get(task_id)
    if status is None:
        return JSONResponse({"error":"unknown task_id"}, status_code=404)
    return JSONResponse({"task_id":task_id, "status":status, "result":res})

@app.post("/cancel/{task_id}")
def cancel_endpoint(task_id: str):
    if task_id not in TASKS:
        return JSONResponse({"error":"unknown task_id"}, status_code=404)
    CANCELLED.add(task_id)
    TASKS[task_id] = "cancelled"
    return JSONResponse({"task_id":task_id, "status":"cancelled"})

@app.post("/cleanup_older")
def cleanup_older(days: int = 1):
    cutoff = time.time() - days*86400
    removed=[]
    for d in UPLOAD_ROOT.iterdir():
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d); removed.append(d.name)
        except Exception:
            pass
    return {"removed":removed}
