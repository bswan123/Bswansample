# server_queue.py
"""
FastAPI server implementing in-memory queue + worker threads + child-process GPT call.

Run with:
  uvicorn server_queue:app --host 0.0.0.0 --port $PORT --workers 1

Env vars (optional):
  OPENAI_API_KEY  - your OpenAI key (recommended)
  WORKER_COUNT    - number of worker threads (default 3)
  CHILD_TIMEOUT   - seconds to wait for child before killing it (default 150)
  UPLOAD_ROOT     - base folder for uploads (default /tmp/uploads)
"""

import os
import uuid
import time
import json
import shutil
import traceback
from pathlib import Path
from multiprocessing import Process
from threading import Thread
import queue
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse

# Try importing openai; keep optional for simulated mode
try:
    import openai
except Exception:
    openai = None

# ---------- Configuration with sensible defaults ----------
UPLOAD_ROOT = Path(os.environ.get("UPLOAD_ROOT", "/tmp/uploads"))
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

WORKER_COUNT = int(os.environ.get("WORKER_COUNT", "3"))
CHILD_TIMEOUT = int(os.environ.get("CHILD_TIMEOUT", "150"))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # if None -> simulated mode

# ---------- In-memory state ----------
TASK_QUEUE = queue.Queue()
TASKS = {}      # task_id -> status: queued | processing | done | failed | cancelled
RESULTS = {}    # task_id -> result dict
CANCELLED = set()
# Map task_id -> child process object so cancel endpoint can terminate running child
PROCESS_MAP = {}

app = FastAPI(title="Exam Solver Queue Server")


# ---------- Utility helpers ----------
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


# ---------- Child process work ----------
def child_process_work(task_id: str, image_paths: List[str], question_number: Optional[str]):
    """
    Runs in a separate process. Writes a JSON file /tmp/uploads/<task_id>/result.json
    with either:
      { "status": "done", "result": {...} }
    or
      { "status": "failed", "error": "..." }
    """
    try:
        # Basic system/user prompt — adapt to your final prompt
        prompt = f"You are an expert exam-solver. There are {len(image_paths)} images for one question."
        if question_number:
            prompt += f" Question number: {question_number}."
        prompt += " Return JSON only: {\"correct_option\":\"A|B|C|D\",\"explanation\":\"...\"}."

        # If no OpenAI key or module available -> simulated response for quick testing
        if openai is None or not OPENAI_API_KEY:
            simulated = {
                "correct_option": "A",
                "explanation": "Simulated result (OPENAI_API_KEY not configured)."
            }
            write_result_file(task_id, {"status": "done", "result": simulated})
            return

        # Real OpenAI usage (example). Adjust model/payload to your account & required model.
        openai.api_key = OPENAI_API_KEY

        system_msg = {
            "role": "system",
            "content": "You are an expert exam-solver that examines images and returns the correct multiple-choice option."
        }
        user_msg = {
            "role": "user",
            "content": prompt + " Do not include any extra commentary; reply with a JSON object."
        }

        # Basic ChatCompletion call (replace model name with one you have access to)
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # change to a model you have access to
            messages=[system_msg, user_msg],
            max_tokens=200,
            temperature=0
        )
        text = resp["choices"][0]["message"]["content"].strip()

        # Attempt to parse JSON; if fails, extract first letter A-D
        parsed = None
        try:
            parsed = json.loads(text)
            # Normalize keys if necessary
            if "correct_option" in parsed:
                result = parsed
            else:
                # fallback to pick A-D if JSON doesn't include correct_option
                import re
                m = re.search(r"[A-D]", text.upper())
                result = {"correct_option": m.group(0) if m else None, "raw": parsed}
        except Exception:
            import re
            m = re.search(r"[A-D]", text.upper())
            result = {"correct_option": m.group(0) if m else None, "raw": text}

        write_result_file(task_id, {"status": "done", "result": result})
    except Exception as e:
        write_result_file(task_id, {"status": "failed", "error": str(e), "trace": traceback.format_exc()})


# ---------- Worker loop ----------
def worker_loop(worker_idx: int):
    print(f"[worker-{worker_idx}] started")
    while True:
        task = TASK_QUEUE.get()
        task_id = task["task_id"]

        if task_id in CANCELLED:
            TASKS[task_id] = "cancelled"
            TASK_QUEUE.task_done()
            print(f"[worker-{worker_idx}] task {task_id} cancelled before start")
            continue

        try:
            TASKS[task_id] = "processing"
            image_paths = task["image_paths"]
            question_number = task.get("question_number")

            p = Process(target=child_process_work, args=(task_id, image_paths, question_number))
            PROCESS_MAP[task_id] = p
            p.start()
            p.join(CHILD_TIMEOUT)

            if p.is_alive():
                print(f"[worker-{worker_idx}] task {task_id} timeout; terminating child")
                p.terminate()
                p.join()
                TASKS[task_id] = "failed"
                RESULTS[task_id] = {"error": "timeout"}
                write_result_file(task_id, {"status": "failed", "error": "timeout"})
            else:
                res = read_result_file(task_id)
                if res is None:
                    TASKS[task_id] = "failed"
                    RESULTS[task_id] = {"error": "no_result_file"}
                else:
                    if res.get("status") == "done":
                        TASKS[task_id] = "done"
                        RESULTS[task_id] = res.get("result")
                    else:
                        TASKS[task_id] = "failed"
                        RESULTS[task_id] = res
        except Exception as e:
            TASKS[task_id] = "failed"
            RESULTS[task_id] = {"error": str(e)}
            print(f"[worker-{worker_idx}] exception: {e}")
        finally:
            PROCESS_MAP.pop(task_id, None)
            TASK_QUEUE.task_done()


# ---------- FastAPI endpoints ----------
@app.on_event("startup")
def startup_event():
    # spawn worker threads
    for i in range(WORKER_COUNT):
        t = Thread(target=worker_loop, args=(i,), daemon=True)
        t.start()
    print(f"Server started with {WORKER_COUNT} worker threads (CHILD_TIMEOUT={CHILD_TIMEOUT}s)")


@app.post("/upload")
async def upload_endpoint(files: List[UploadFile] = File(...), batch_id: Optional[str] = Form(None),
                          question_number: Optional[str] = Form(None)):
    task_id = str(uuid.uuid4())
    try:
        paths = save_upload_files(task_id, files)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed saving files: {e}")

    TASKS[task_id] = "queued"
    TASK_QUEUE.put({"task_id": task_id, "image_paths": paths, "batch_id": batch_id, "question_number": question_number})

    return JSONResponse({"task_id": task_id, "status": "queued"})


@app.get("/result/{task_id}")
def result_endpoint(task_id: str):
    status = TASKS.get(task_id)
    res = RESULTS.get(task_id)
    if status is None:
        return JSONResponse({"error": "unknown task_id"}, status_code=404)
    return JSONResponse({"task_id": task_id, "status": status, "result": res})


@app.post("/cancel/{task_id}")
def cancel_endpoint(task_id: str):
    if task_id not in TASKS:
        return JSONResponse({"error": "unknown task_id"}, status_code=404)

    # If queued, mark cancelled; if processing, try to terminate child
    CANCELLED.add(task_id)
    TASKS[task_id] = "cancelled"
    p = PROCESS_MAP.get(task_id)
    if p is not None and p.is_alive():
        try:
            p.terminate()
            p.join(timeout=5)
            write_result_file(task_id, {"status": "failed", "error": "cancelled_by_user"})
            RESULTS[task_id] = {"error": "cancelled_by_user"}
        except Exception:
            pass
    return JSONResponse({"task_id": task_id, "status": "cancelled"})


@app.get("/test")
def test_html():
    html = """
    <html>
      <head><title>Upload Test</title></head>
      <body>
        <h3>Manual upload test</h3>
        <form action="/upload" enctype="multipart/form-data" method="post">
          Batch ID (optional): <input name="batch_id"/><br/><br/>
          Question number (optional): <input name="question_number"/><br/><br/>
          Images: <input name="files" type="file" multiple/><br/><br/>
          <input type="submit" value="Upload">
        </form>
        <p>After upload you'll get a task_id JSON response. Poll <code>/result/&lt;task_id&gt;</code> to see status.</p>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.post("/cleanup_older")
def cleanup_older(days: int = 1):
    cutoff = time.time() - days * 86400
    removed = []
    for d in UPLOAD_ROOT.iterdir():
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d)
                removed.append(d.name)
        except Exception:
            pass
    return {"removed": removed}


# For local run convenience
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_queue:app", host="0.0.0.0", port=8000, workers=1)
