import os, json, time, uuid, shutil, tempfile
import cv2, numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from . import db, engine

db.init()
app = FastAPI(title="Helmet Violation Detection API", version="1.0",
              description="YOLO detection + ByteTrack tracking + SQLite violation log.")
app.mount("/results", StaticFiles(directory=str(engine.RESULTS)), name="results")


def decode(data: bytes):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image file")
    return img


def with_urls(events):
    return [{**e, "snapshot_url": f"/results/{e['snapshot']}" if e.get("snapshot") else None} for e in events]


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(engine.ROOT / "app" / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "model": engine.MODEL_PATH.name, "classes": engine.get_model().names,
            "violation_class": engine.VIOLATION_CLASS, "min_hits": engine.MIN_HITS}


@app.post("/api/detect/image")
async def detect_image(file: UploadFile = File(...), conf: float = Query(0.35, ge=0.05, le=0.95), imgsz: int = 640):
    img = decode(await file.read())
    dets, counts, ms = await run_in_threadpool(engine.detect_image, img, conf, imgsz)
    out = f"{uuid.uuid4().hex[:10]}.jpg"
    cv2.imwrite(str(engine.RESULTS / out), engine.annotate(img, dets))
    viol = counts.get(engine.VIOLATION_CLASS, 0)
    heads = sum(counts.values())
    events = [{"class": d["class"], "confidence": d["confidence"], "frame": 0, "t_sec": 0.0, "snapshot": out}
              for d in dets if d["class"] == engine.VIOLATION_CLASS]
    run_id = db.save_run("image", file.filename, conf, 1, None, ms, heads, viol, heads - viol, out, events)
    return {"run_id": run_id, "counts": counts, "riders": heads, "violations": viol, "compliant": heads - viol,
            "processing_ms": ms, "detections": dets, "image_url": f"/results/{out}"}


@app.post("/api/detect/video")
async def detect_video(file: UploadFile = File(...), conf: float = Query(0.35, ge=0.05, le=0.95), imgsz: int = 640):
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        await run_in_threadpool(shutil.copyfileobj, file.file, f)
        src = f.name
    out = f"{uuid.uuid4().hex[:10]}.mp4"
    try:
        res = await run_in_threadpool(engine.track_video, src, engine.RESULTS / out, conf, imgsz)
    finally:
        os.remove(src)
    if res["frames"] == 0:
        raise HTTPException(400, "Could not read any frames from this video")
    run_id = db.save_run("video", file.filename, conf, res["frames"], res["fps"], res["processing_ms"],
                         res["riders"], res["violations"], res["compliant"], out, res["events"])
    res["events"] = with_urls(res["events"])
    return {"run_id": run_id, **res, "video_url": f"/results/{out}"}


@app.get("/api/runs")
def runs(limit: int = 50):
    return db.list_runs(min(max(limit, 1), 500))


@app.get("/api/events")
def events(run_id: int | None = None, limit: int = 100):
    return with_urls(db.list_events(run_id, min(max(limit, 1), 500)))


@app.get("/api/stats")
def stats():
    return db.stats()


@app.delete("/api/history")
def clear_history():
    files = db.clear()
    for f in set(files):
        try: (engine.RESULTS / f).unlink()
        except OSError: pass
    return {"deleted_files": len(set(files))}


@app.websocket("/ws")
async def ws_live(ws: WebSocket):
    """Webcam: client sends JPEG bytes (+ optional JSON {conf, imgsz}); server returns tracked boxes + violation summary."""
    await ws.accept()
    m = await run_in_threadpool(engine.new_tracker_model)
    conf, imgsz, frames, t0 = 0.35, 416, 0, time.perf_counter()
    vt = engine.ViolationTracker(fps=15.0)
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("text"):
                cfg = json.loads(msg["text"])
                conf = min(max(float(cfg.get("conf", conf)), 0.05), 0.95)
                imgsz = int(cfg.get("imgsz", imgsz)); imgsz = imgsz if imgsz in (320, 416, 640) else 416
                continue
            data = msg.get("bytes")
            if not data:
                continue
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            t = time.perf_counter()
            dets = await run_in_threadpool(engine.track_frame, m, img, conf, imgsz)
            ms = (time.perf_counter() - t) * 1000
            new = vt.update(frames, img, dets); frames += 1
            h, w = img.shape[:2]
            await ws.send_json({"ms": round(ms, 1), "w": w, "h": h, "detections": dets,
                                "summary": vt.summary(), "new_violations": len(new)})
    except WebSocketDisconnect:
        pass
    finally:
        if frames > 0:
            el, s = time.perf_counter() - t0, vt.summary()
            db.save_run("webcam", None, conf, frames, round(frames / el, 1), round(el * 1000),
                        s["riders"], s["violations"], s["compliant"], None, vt.events)
