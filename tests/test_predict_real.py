import os
import pytest
import io
from PIL import Image
import numpy as np

REAL_TEST = os.getenv("REAL_YOLO_TEST", "0") == "1"

@pytest.mark.skipif(not REAL_TEST, reason="Set REAL_YOLO_TEST=1 to run real model test.")
def test_predict_real_weights(client):
    # make a dummy image (or load a known test sample if you prefer)
    arr = (np.zeros((320, 320, 3)) + 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)

    files = {"file": ("white.png", buf.getvalue(), "image/png")}
    r = client.post("/predict?conf=0.25", files=files, timeout=60)
    assert r.status_code == 200
    body = r.json()
    assert "boxes" in body and "latency_ms" in body
