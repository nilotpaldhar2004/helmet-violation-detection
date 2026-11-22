import os, tempfile, pathlib
os.environ["DB_PATH"] = str(pathlib.Path(tempfile.mkdtemp()) / "test.db")
os.environ["RESULTS_DIR"] = tempfile.mkdtemp()

from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)
SAMPLE = pathlib.Path(__file__).parent / "sample.jpg"


def post_image(**params):
    with open(SAMPLE, "rb") as f:
        return client.post("/api/detect/image", params=params, files={"file": ("sample.jpg", f, "image/jpeg")})


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and "no_helmet" in body["classes"].values()


def test_detect_image_schema():
    r = post_image(conf=0.25)
    assert r.status_code == 200
    d = r.json()
    for key in ("counts", "riders", "violations", "compliant", "processing_ms", "detections", "image_url"):
        assert key in d
    assert d["riders"] == d["violations"] + d["compliant"]
    assert d["processing_ms"] > 0


def test_bad_image_returns_400():
    r = client.post("/api/detect/image", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400


def test_confidence_validation():
    assert post_image(conf=2.0).status_code == 422


def test_history_and_events_are_stored():
    post_image(conf=0.25)
    runs = client.get("/api/runs").json()
    assert len(runs) >= 1 and runs[0]["source"] == "image"
    assert isinstance(client.get("/api/events").json(), list)
    assert client.get("/api/stats").json()["runs"] >= 1


def test_websocket_roundtrip():
    jpg = SAMPLE.read_bytes()
    with client.websocket_connect("/ws") as ws:
        ws.send_bytes(jpg)
        data = ws.receive_json()
    assert "detections" in data and "summary" in data
