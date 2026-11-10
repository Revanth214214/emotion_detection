"""API tests. The real model is replaced with a stub so no weights are needed."""
import sys
import types

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Stub the pipeline module before app.py imports it (avoids loading torch/model).
fake_mod = types.ModuleType("src.inference.pipeline.inference_pipeline")


class FakePipeline:
    device = "cpu"
    faces = [{
        "emotion": "Happy", "top_emotion": "Happy", "confidence": 0.9,
        "probabilities": {"Happy": 0.9}, "box": [1, 2, 3, 4],
    }]

    def run_frame(self, frame, min_confidence=0.5):
        return self.faces


fake_mod.InferencePipeline = FakePipeline
for name in ("src", "src.inference", "src.inference.pipeline"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["src.inference.pipeline.inference_pipeline"] = fake_mod

import app as app_module  # noqa: E402

client = TestClient(app_module.app)


def _jpeg(size=64):
    ok, buf = cv2.imencode(".jpg", np.zeros((size, size, 3), np.uint8))
    assert ok
    return buf.tobytes()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "healthy"


def test_predict_success():
    r = client.post("/predict", files={"file": ("f.jpg", _jpeg(), "image/jpeg")})
    body = r.json()
    assert r.status_code == 200
    assert body["status"] == "success" and body["num_faces"] == 1
    assert body["prediction"][0]["emotion"] == "Happy"


def test_predict_no_face(monkeypatch):
    monkeypatch.setattr(app_module.pipeline, "faces", [], raising=False)
    r = client.post("/predict", files={"file": ("f.jpg", _jpeg(), "image/jpeg")})
    assert r.json()["status"] == "no_face"


def test_rejects_non_image_type():
    r = client.post("/predict", files={"file": ("a.txt", b"hi", "text/plain")})
    assert r.status_code == 415


def test_rejects_corrupt_image():
    r = client.post("/predict", files={"file": ("f.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400


def test_rejects_oversized(monkeypatch):
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 10)
    r = client.post("/predict", files={"file": ("f.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 413
