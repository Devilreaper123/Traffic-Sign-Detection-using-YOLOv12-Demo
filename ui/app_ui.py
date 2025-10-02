# app_ui.py
import io
import os
import time
from typing import List, Dict, Any

import numpy as np
import requests
import streamlit as st
from PIL import Image

# ───────────── Config ─────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")
DEFAULT_CONF = float(os.getenv("DEFAULT_CONF", "0.25"))
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "200"))

SESSION = requests.Session()
MAX_SIZE = 640  # resize before upload


# ───────────── API CALLS ─────────────
def call_predict_api(img: Image.Image, conf: float) -> Dict[str, Any]:
    """Call single-image /predict endpoint."""
    img = img.copy()
    img.thumbnail((MAX_SIZE, MAX_SIZE))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    buf.seek(0)
    files = {"file": ("upload.jpg", buf.getvalue(), "image/jpeg")}

    t0 = time.perf_counter()
    r = SESSION.post(
        f"{API_URL}/predict", params={"conf": conf}, files=files, timeout=30
    )
    latency_client_ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    data = r.json()
    data["latency_client_ms"] = round(latency_client_ms, 2)
    return data
def get_api_workers(default: int = 1) -> int:
    try:
        r = SESSION.get(f"{API_URL}/info", timeout=5)
        r.raise_for_status()
        return max(1, int(r.json().get("workers", default)))
    except Exception:
        return default


def call_predict_batch(imgs: List[Image.Image], conf: float) -> Dict[str, Any]:
    """Call /predict_batch endpoint with multiple images."""
    files = []
    for i, im in enumerate(imgs):
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        files.append(("files", (f"img_{i}.jpg", buf.getvalue(), "image/jpeg")))

    t0 = time.perf_counter()
    r = SESSION.post(
        f"{API_URL}/predict_batch", params={"conf": conf}, files=files, timeout=60
    )
    elapsed = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    data = r.json()
    data["latency_client_ms"] = round(elapsed, 2)
    return data


def warmup():
    try:
        r = requests.post(f"{API_URL}/warmup", timeout=30)
        return r.ok, (
            r.json()
            if r.headers.get("content-type", "").startswith("application/json")
            else r.text
        )
    except Exception as e:
        return False, str(e)


# ───────────── Utils ─────────────
def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    arr = np.array(xs, dtype=float)
    return float(np.percentile(arr, p))


def throughput(images: int, elapsed_s: float) -> float:
    if elapsed_s <= 0:
        return float("nan")
    return images / elapsed_s


from concurrent.futures import ThreadPoolExecutor, as_completed


def run_benchmark_parallel(imgs, conf=0.25, max_workers=4):
    latencies, errors = [], 0
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(call_predict_api, im, conf) for im in imgs]
        for fut in as_completed(futures):
            try:
                d = fut.result()
                latencies.append(float(d.get("latency_ms", 0.0)))
            except Exception:
                errors += 1
    elapsed = time.perf_counter() - start
    tput = throughput(len(imgs) - errors, elapsed)
    return latencies, errors, elapsed, tput


# ───────────── UI ─────────────
st.set_page_config(page_title="YOLOv12m API Benchmark", layout="wide")
st.title("YOLOv12m Traffic Sign Detection — API Demo + Benchmarks")
st.caption("Tracking: MLflow at http://localhost:5000 (in Docker: http://mlflow:5000)")
if "api_workers" not in st.session_state:
    st.session_state.api_workers = get_api_workers()
with st.sidebar:
    st.subheader("Connection")
    st.text_input("API URL", value=API_URL, key="api_url_box", disabled=True)
    
    st.write(f"Detected API workers: **{st.session_state.api_workers}**")
    if st.button("Warmup model on API"):
        ok, msg = warmup()
        st.success("Warmup OK: Model Loaded" if ok else "Warmup failed")

    st.subheader("Inference Parameters")
    conf = st.slider("Confidence threshold", 0.05, 0.95, DEFAULT_CONF, 0.05)


if "history" not in st.session_state:
    st.session_state.history = []


colL, colR = st.columns([0.55, 0.45], gap="large")

# ─── Left: Single image demo ───
with colL:
    st.header("Single Image Inference")
    uploaded = st.file_uploader("Upload JPG/PNG", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        try:
            img = Image.open(uploaded).convert("RGB")
        except Exception as e:
            st.error(f"Could not open image: {e}")
            st.stop()

        st.image(img, caption="Input Image", use_column_width=True)

        with st.spinner("Calling API…"):
            try:
                resp = call_predict_api(img, conf=conf)
                boxes = resp.get("boxes", [])
                server_ms = float(resp.get("latency_ms", float("nan")))
                client_ms = float(resp.get("latency_client_ms", float("nan")))

                st.session_state.history.append(
                    {
                        "server_ms": server_ms,
                        "client_ms": client_ms,
                        "ok": True,
                        "ts": time.strftime("%H:%M:%S"),
                    }
                )
                st.session_state.history = st.session_state.history[-MAX_HISTORY:]

                st.success(
                    f"Server latency: {server_ms:.1f} ms | End-to-end: {client_ms:.1f} ms"
                )

                if boxes:
                    st.subheader("Detections")
                    table_data = [
                        {"Class": b["cls"], "Conf": f"{b['conf']:.2f}"} for b in boxes
                    ]
                    st.table(table_data)
                else:
                    st.info("No detections returned by API.")
            except Exception as e:
                st.session_state.history.append({"ok": False})
                st.error(f"API call failed: {e}")


# ─── Right: Metrics + Benchmarks ───
with colR:
    st.header("Ops Metrics (rolling)")
    hist = st.session_state.history
    ok_count = sum(1 for h in hist if h.get("ok"))
    err_count = len(hist) - ok_count
    server_lat = [h.get("server_ms") for h in hist if h.get("ok")]
    client_lat = [h.get("client_ms") for h in hist if h.get("ok")]

    c1, c2, c3 = st.columns(3)
    c1.metric("Requests", len(hist))
    c2.metric("Errors", err_count)
    c3.metric("Success rate", f"{(100.0*ok_count/max(1,len(hist))):.1f}%")

    c4, c5, c6 = st.columns(3)
    c4.metric(
        "p50 server (ms)", f"{percentile(server_lat,50):.1f}" if server_lat else "–"
    )
    c5.metric(
        "p95 server (ms)", f"{percentile(server_lat,95):.1f}" if server_lat else "–"
    )
    c6.metric(
        "p50 end-to-end (ms)", f"{percentile(client_lat,50):.1f}" if client_lat else "–"
    )

    # Benchmark
    st.subheader("Benchmark")
    mode = st.selectbox("Mode", ["Sequential", "Parallel", "Batch"])
    n = st.slider("Number of requests (batch size)", 1, 30, 5)
    run_btn = st.button("Run benchmark")

    if run_btn:
        imgs = (
            [Image.open(uploaded).convert("RGB")] * n
            if uploaded
            else [
                Image.fromarray(np.full((320, 320, 3), 220, dtype=np.uint8))
                for _ in range(n)
            ]
        )

        if mode == "Sequential":
            start = time.perf_counter()
            latencies, errors = [], 0
            for im in imgs:
                try:
                    d = call_predict_api(im, conf=conf)
                    latencies.append(float(d.get("latency_ms", 0.0)))
                except Exception:
                    errors += 1
            elapsed = time.perf_counter() - start
            tput = throughput(n - errors, elapsed)

        elif mode == "Parallel":
            latencies, errors, elapsed, tput = run_benchmark_parallel(
                imgs, conf=conf, max_workers=4
            )

        else:  # Batch
            resp = call_predict_batch(imgs, conf=conf)
            results = resp.get("results", [])
            latencies = [float(x.get("latency_ms", 0.0)) for x in results]
            errors = 0
            elapsed = resp.get("batch_latency_ms", 0) / 1000.0
            tput = throughput(len(results), elapsed)

        # Results
        st.write(f"**Mode:** {mode} | **Batch size:** {n}")
        st.write(
            f"**Total time:** {elapsed:.2f}s | **Throughput:** {tput:.2f} img/s | **Errors:** {errors}"
        )
        if latencies:
            st.write(
                f"**Server p50:** {percentile(latencies,50):.1f} ms | **p95:** {percentile(latencies,95):.1f} ms"
            )

st.divider()
st.caption(
    "This UI calls the FastAPI model service. Shows image + detection table and benchmarks with Sequential, Parallel, or Batch modes."
)
