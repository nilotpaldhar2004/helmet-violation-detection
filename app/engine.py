"""Detection, ByteTrack tracking and violation logic."""
import os, time, uuid, shutil, subprocess, threading
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "best.pt"))
RESULTS = Path(os.getenv("RESULTS_DIR", ROOT / "results"))
(RESULTS / "snapshots").mkdir(parents=True, exist_ok=True)
VIOLATION_CLASS = os.getenv("VIOLATION_CLASS", "no_helmet")
MIN_HITS = int(os.getenv("MIN_HITS", "5"))          # frames a track must show 'no_helmet' to be confirmed
MAX_FRAMES = int(os.getenv("MAX_VIDEO_FRAMES", "1500"))
FONT = cv2.FONT_HERSHEY_SIMPLEX

_model, _lock = None, threading.Lock()


def get_model():
    global _model
    if _model is None:
        _model = YOLO(str(MODEL_PATH))
    return _model


def new_tracker_model():
    """Fresh instance = fresh ByteTrack state (IDs restart at 1)."""
    m = YOLO(str(MODEL_PATH))
    m.predict(np.zeros((64, 64, 3), "uint8"), verbose=False)
    return m


def color_for(cls):
    return (0, 0, 255) if cls == VIOLATION_CLASS else (0, 200, 0)      # BGR: red violation, green OK


def annotate(img, dets, hud=None):
    out = img.copy()
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d["box_xyxy"]); col = color_for(d["class"])
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        label = (f"#{d['id']} " if d.get("id") is not None else "") + f"{d['class']} {d['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        top = max(y1 - th - 8, 0)
        cv2.rectangle(out, (x1, top), (x1 + tw + 6, top + th + 8), col, -1)
        cv2.putText(out, label, (x1 + 3, top + th + 3), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if hud:
        (tw, th), _ = cv2.getTextSize(hud, FONT, 0.6, 2)
        cv2.rectangle(out, (0, 0), (tw + 16, th + 16), (0, 0, 0), -1)
        cv2.putText(out, hud, (8, th + 6), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def to_dets(r, names):
    ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(r.boxes)
    return [{"id": i, "class": names[int(c)], "confidence": round(float(s), 3),
             "box_xyxy": [round(float(v), 1) for v in b]}
            for i, c, s, b in zip(ids, r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]


def detect_image(img, conf, imgsz=640):
    m = get_model()
    t = time.perf_counter()
    with _lock:
        r = m.predict(img, conf=conf, imgsz=imgsz, verbose=False)[0]
    ms = round((time.perf_counter() - t) * 1000, 1)
    dets = to_dets(r, m.names)
    counts = dict(Counter(d["class"] for d in dets))
    return dets, counts, ms


def track_frame(m, img, conf, imgsz):
    r = m.track(img, persist=True, tracker="bytetrack.yaml", conf=conf, imgsz=imgsz, verbose=False)[0]
    return to_dets(r, m.names)


def save_snapshot(frame, box, tid, frame_idx):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    pw, ph = (x2 - x1) * 0.4, (y2 - y1) * 0.4
    cx1, cy1, cx2, cy2 = int(max(x1 - pw, 0)), int(max(y1 - ph, 0)), int(min(x2 + pw, w)), int(min(y2 + ph, h))
    crop = frame[cy1:cy2, cx1:cx2].copy()
    cv2.rectangle(crop, (int(x1 - cx1), int(y1 - cy1)), (int(x2 - cx1), int(y2 - cy1)), (0, 0, 255), 2)
    name = f"snapshots/{uuid.uuid4().hex[:10]}.jpg"
    cv2.imwrite(str(RESULTS / name), crop)
    return name


class ViolationTracker:
    """Counts each tracked rider once; logs a violation when no_helmet is seen consistently."""

    def __init__(self, fps=25.0):
        self.tracks, self.events, self.fps = {}, [], fps

    def update(self, frame_idx, frame, dets):
        new = []
        for d in dets:
            tid = d["id"]
            if tid is None:
                continue
            t = self.tracks.setdefault(tid, {"obs": 0, "viol": 0, "logged": False})
            t["obs"] += 1
            if d["class"] == VIOLATION_CLASS:
                t["viol"] += 1
            if not t["logged"] and t["viol"] >= MIN_HITS and t["viol"] / t["obs"] >= 0.5:
                t["logged"] = True
                ev = {"event_type": "no_helmet_violation", "track_id": tid, "class": d["class"],
                      "confidence": d["confidence"], "frame": frame_idx,
                      "t_sec": round(frame_idx / self.fps, 2),
                      "snapshot": save_snapshot(frame, d["box_xyxy"], tid, frame_idx)}
                self.events.append(ev); new.append(ev)
        return new

    def summary(self):
        confirmed = [t for t in self.tracks.values() if t["obs"] >= MIN_HITS]
        violations = sum(1 for t in confirmed if t["logged"])
        riders = len(confirmed)
        return {"riders": riders, "violations": violations, "compliant": max(riders - violations, 0),
                "compliance_rate": round((riders - violations) / riders, 3) if riders else None}


def track_video(src, dst, conf, imgsz=640):
    m = new_tracker_model()
    dst = Path(dst)
    cap = cv2.VideoCapture(str(src))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw = dst.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h))
    vt, n, t0 = ViolationTracker(fps_src), 0, time.perf_counter()
    while n < MAX_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        dets = track_frame(m, frame, conf, imgsz)
        vt.update(n, frame, dets)
        n += 1
        s = vt.summary()
        hud = f"Violations: {s['violations']}  Riders: {s['riders']}  {n / (time.perf_counter() - t0):.0f} FPS"
        vw.write(annotate(frame, dets, hud))
    cap.release(); vw.release()
    elapsed = time.perf_counter() - t0
    if n == 0:
        raw.unlink(missing_ok=True)
        return {"frames": 0}
    if shutil.which("ffmpeg"):            # H.264 so browsers can play it
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-vcodec", "libx264",
                        "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-movflags", "+faststart", str(dst)], check=False)
        raw.unlink(missing_ok=True)
    if not dst.exists():
        raw.rename(dst)
    return {"frames": n, "fps": round(n / elapsed, 1), "processing_ms": round(elapsed * 1000),
            **vt.summary(), "events": vt.events, "truncated": n >= MAX_FRAMES}
