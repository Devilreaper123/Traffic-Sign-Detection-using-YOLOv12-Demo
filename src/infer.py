import io
import time
from typing import List, Tuple
import numpy as np
from PIL import Image
import cv2

from ultralytics import YOLO
from .config import WEIGHTS_PATH, INPUT_SIZE, CLASS_NAMES

_model = None  # lazy singleton


def get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(str(WEIGHTS_PATH))
    return _model


def _read_image_bytes(file_bytes: bytes) -> np.ndarray:
    # PIL -> RGB np array
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.array(img)


def _resize_if_needed(img: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    # Keep simple square resize to training size (you can swap to letterbox if desired)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def predict_ndarray(img: np.ndarray, conf: float = 0.25) -> Tuple[List[dict], float]:
    """
    Runs prediction on an RGB ndarray image and returns
    - boxes: list of dicts {cls, conf, x1,y1,x2,y2}
    - latency_ms
    """
    model = get_model()
    pre_t = time.time()
    resized = _resize_if_needed(img, INPUT_SIZE)
    # Ultralytics expects BGR or RGB depending on API; ndarray RGB is OK
    t0 = time.time()
    results = model.predict(resized, conf=conf, verbose=False)
    inf_ms = (time.time() - t0) * 1000.0

    boxes = []
    if len(results) > 0:
        r = results[0]
        if r.boxes is not None and r.boxes.xyxy is not None:
            xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy().astype(float)
            for (x1, y1, x2, y2), c_id, c in zip(xyxy, cls_ids, confs):
                name = CLASS_NAMES[c_id] if 0 <= c_id < len(CLASS_NAMES) else str(c_id)
                boxes.append({
                    "cls": name,
                    "conf": float(round(c, 4)),
                    "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                })

    total_ms = (time.time() - pre_t) * 1000.0
    # You can also return both preprocess + inference + postprocess times if you want
    return boxes, total_ms


def predict_file(file_bytes: bytes, conf: float = 0.25):
    img = _read_image_bytes(file_bytes)
    return predict_ndarray(img, conf=conf)
