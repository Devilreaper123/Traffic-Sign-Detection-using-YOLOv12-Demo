from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from typing import Optional, List
from contextlib import asynccontextmanager
import traceback, time, os, csv

from prometheus_fastapi_instrumentator import Instrumentator
from .schemas import Prediction, Health, Warmup
from .infer import predict_file, get_model
from .mltrack import (
    start as ml_start,
    log_async as ml_log,
)


# ───────────── Lifespan (startup/shutdown) ─────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_start()  # starts async MLflow logger worker (ok in lifespan)
    yield
    # (optional) stop/flush your worker here if you expose a stop()


# ───────────── App setup ─────────────
app = FastAPI(title="YOLOv12m Traffic Sign API", version="1.0.0", lifespan=lifespan)

# ───────────── App setup ─────────────
instrumentator = Instrumentator().instrument(app)
instrumentator.expose(app, endpoint="/metrics", include_in_schema=False)
# CORS for UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust to ["https://demo.yourdomain.com"] in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip responses
app.add_middleware(GZipMiddleware, minimum_size=500)


# ───────────── Endpoints ─────────────
@app.get("/info")
def info():
    workers = int(os.getenv("API_WORKERS", os.getenv("WORKERS", "1")))
    return {"name": "yolo-traffic-sign-api", "version": "1.0.0", "workers": workers}


@app.get("/healthz", response_model=Health, tags=["health"])
def health() -> Health:
    return Health(status="ok")


@app.post("/warmup", response_model=Warmup, tags=["health"])
def warmup() -> Warmup:
    """Loads the model once to remove cold start penalty."""
    try:
        _ = get_model()
        return Warmup(ok=True, msg="Model loaded")
    except Exception as e:
        return Warmup(ok=False, msg=f"Warmup failed: {e}")


@app.post("/predict", tags=["inference"])
async def predict(file: UploadFile = File(...), conf: Optional[float] = 0.25):
    if file.content_type is None or not file.content_type.startswith(
        ("image/", "application/octet-stream")
    ):
        raise HTTPException(status_code=400, detail="Please upload an image file.")
    try:
        img_bytes = await file.read()
        boxes, latency_ms = predict_file(img_bytes, conf=conf)

        # CSV logging
        LOG_PATH = os.getenv("PRED_LOG", "artifacts/predict_log.csv")
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for b in boxes:
                w.writerow(
                    [
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        file.filename,
                        b["cls"],
                        b["conf"],
                        b["x1"],
                        b["y1"],
                        b["x2"],
                        b["y2"],
                        round(latency_ms, 3),
                    ]
                )

        # MLflow async logging
        counts = {}
        for b in boxes:
            counts[b["cls"]] = counts.get(b["cls"], 0) + 1

        ml_log(
            run_name="inference",
            metrics={
                "latency_ms": float(latency_ms),
                "n_boxes": float(len(boxes)),
                **{f"class_{k}_count": float(v) for k, v in counts.items()},
            },
            params={
                "conf": str(conf),
                "imgsz": "640",
                "api_workers": os.getenv("API_WORKERS", "1"),
            },
        )

        return JSONResponse(
            content={
                "file": file.filename,
                "conf_threshold": conf,
                "boxes": boxes,
                "latency_ms": round(latency_ms, 3),
            }
        )
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/predict_batch", tags=["inference"])
async def predict_batch(files: List[UploadFile] = File(...), conf: float = 0.25):
    """Run batch inference on multiple uploaded images."""
    results = []
    t0 = time.perf_counter()

    total_counts = {}
    total_boxes = 0

    for f in files:
        bts = await f.read()
        boxes, latency_ms = predict_file(bts, conf=conf)
        results.append(
            {"filename": f.filename, "boxes": boxes, "latency_ms": latency_ms}
        )

        # accumulate counts
        for b in boxes:
            total_boxes += 1
            total_counts[b["cls"]] = total_counts.get(b["cls"], 0) + 1

    elapsed = (time.perf_counter() - t0) * 1000.0
    avg_latency = elapsed / max(1, len(files))

    # MLflow async logging for batch
    ml_log(
        run_name="batch_inference",
        metrics={
            "batch_latency_ms": round(elapsed, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "batch_size": len(files),
            "n_boxes_total": float(total_boxes),
            **{f"class_{k}_count": float(v) for k, v in total_counts.items()},
        },
        params={
            "conf": str(conf),
            "imgsz": "640",
            "api_workers": os.getenv("API_WORKERS", "1"),
        },
    )

    return {
        "batch_size": len(files),
        "results": results,
        "batch_latency_ms": round(elapsed, 2),
        "avg_latency_ms": round(avg_latency, 2),
    }
