import io
from PIL import Image
import numpy as np

def make_png_bytes(w=320, h=320):
    # simple RGB image in memory
    arr = (np.zeros((h, w, 3)) + 255).astype("uint8")  # white
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

def test_predict_stubbed(client, monkeypatch):
    def fake_predict_file(bts, conf=0.25):
        return ([{
            "cls": "Danger Ahead",
            "conf": 0.99,
            "x1": 10, "y1": 10, "x2": 100, "y2": 100
        }], 5.123)

    # 🔁 Patch the name used by the API module, not the source module
    import src.service as service
    monkeypatch.setattr(service, "predict_file", fake_predict_file)

    img_bytes = make_png_bytes()
    files = {"file": ("00006.png", img_bytes, "image/png")}
    r = client.post("/predict?conf=0.25", files=files)
    assert r.status_code == 200

    body = r.json()
    assert isinstance(body["boxes"], list) and len(body["boxes"]) == 1
    b = body["boxes"][0]
    assert b["cls"] == "Danger Ahead"
    assert 0 <= b["conf"] <= 1
    assert body["latency_ms"] >= 0

