# 🚦 Traffic Sign Detection using YOLOv12m

End-to-end **Dockerized YOLOv12m pipeline** for real-time traffic sign detection.  
Includes **FastAPI inference service**, **Streamlit UI**, **MLflow experiment tracking**, and **Prometheus metrics**, orchestrated with **Docker Compose** (CPU/GPU ready).

---

## 📌 Architecture

```
                 ┌────────────────────────────┐
                 │         Streamlit UI       │
                 │   (Latency/Throughput UX)  │
                 └─────────────▲──────────────┘
                               │ API calls
                               │
                 ┌─────────────┴──────────────┐
                 │         FastAPI API        │
                 │   /predict & /predict_batch│
                 └─────────────▲──────────────┘
                               │ logs metrics
                               │
                 ┌─────────────┴──────────────┐
                 │           MLflow           │
                 │ (Experiments, Metrics)     │
                 └─────────────▲──────────────┘
                               │ exposes /metrics
                               │
                 ┌─────────────┴──────────────┐
                 │       Prometheus (opt.)    │
                 │   Scrapes API metrics      │
                 └────────────────────────────┘
```

---

## ✅ Features

- 🚀 **FastAPI Inference API** (`/predict`, `/predict_batch`)  
- 📊 **Streamlit UI** for benchmarking & visualization  
- 🧪 **MLflow** experiment tracking (latency, throughput, class counts)  
- 🐳 **Docker Compose** orchestration (CPU/GPU profiles)  
- 📉 **Prometheus Metrics** at `/metrics`  
- 🔐 Environment-variable configs for flexible deployments  
- 📂 Logs & predictions saved under `/app/artifacts`  

---

## ⚡ Benchmarks (YOLOv12m, imgsz=640, 2 workers, Dockerized)

- **Single image:** ~121 ms end-to-end (server ~104 ms)  
- **Sequential (30 imgs):** 6.4 img/s, p50=99 ms, p95=124 ms  
- **Parallel (30 imgs, 2 workers):** 14.8 img/s, p50=106 ms, p95=133 ms  
- **Batch (30 imgs):** 14.2 img/s, p50=70 ms, p95=81 ms  

---

## 🛠️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/<yourname>/Traffic-Sign-Detection-using-YOLOv12-Demo.git
cd Traffic-Sign-Detection-using-YOLOv12-Demo
```

### 2. Build & run with Docker Compose
```bash
docker compose up -d --build
```

### 3. Access services
- API → [http://localhost:8000/docs](http://localhost:8000/docs)  
- UI → [http://localhost:8501](http://localhost:8501)  
- MLflow → [http://localhost:5000](http://localhost:5000)  
- Prometheus (if enabled) → [http://localhost:9090](http://localhost:9090)  

### 4. Test API
```bash
# Health check
curl http://localhost:8000/healthz

# Warmup
curl -X POST http://localhost:8000/warmup

# Single image predict
curl -X POST "http://localhost:8000/predict?conf=0.25"   -H "Content-Type: multipart/form-data"   -F "file=@samples/sign.jpg"
```

---

## 🐳 Docker Cheat Sheet

```bash
# List containers
docker ps -a

# Logs
docker logs yolo-api

# Exec into container
docker exec -it yolo-api sh

# Stop & remove
docker stop yolo-api && docker rm yolo-api

# Clean everything
docker system prune -af
docker rmi $(docker images -aq)

# Manual build + push
docker build -t <dockerhub-user>/traffic-sign-yolo:latest -f Dockerfile .
docker push <dockerhub-user>/traffic-sign-yolo:latest
```

---

## 📂 Repository Layout

```
.
├── src/
│   ├── service.py       # FastAPI app with endpoints
│   ├── infer.py         # Model loading + inference
│   ├── schemas.py       # Pydantic models
│   ├── mltrack.py       # Async MLflow logging
├── ui/
│   ├── app_ui.py        # Streamlit benchmarking UI
│   └── Dockerfile.ui
├── models/              # Stores best.pt (mounted/baked)
├── artifacts/           # Logs, CSVs
├── docker-compose.yml
├── Dockerfile           # API (CPU/GPU support)
└── README.md
```

---

## 📊 MLflow Integration

- Tracks automatically:
  - `latency_ms`, `batch_latency_ms`, `avg_latency_ms`  
  - `n_boxes`, per-class counts  
- Configured via environment:
  ```yaml
  environment:
    MLFLOW_TRACKING_URI: "http://mlflow:5000"
    MLFLOW_EXPERIMENT_NAME: "yolov12m-traffic-sign"
  ```
- UI available at → [http://localhost:5000](http://localhost:5000)

---

## 🚀 CI/CD (GitHub Actions)

GitHub Actions builds & pushes images to DockerHub:

`.github/workflows/docker-ci.yml`
```yaml
name: CI/CD Docker

on:
  push:
    branches: [ main ]

jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: docker/setup-buildx-action@v2
      - uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
      - uses: docker/build-push-action@v4
        with:
          push: true
          tags: <dockerhub-user>/traffic-sign-yolo:latest
```

---

## 🌐 Deployment

### Local
Runs via `docker compose up` (CPU/GPU profiles supported).

### AWS EC2
- Pull image from DockerHub or your ECR registry.  
- Run with `docker compose` or plain `docker run`.  
- Open inbound ports: **8000 (API)**, **8501 (UI)**, **5000 (MLflow)**.  

### AWS ECS (Fargate)
- Task definition with 3 services: API, UI, MLflow.  
- Attach security group with ports `8000, 8501, 5000`.  
- Optionally use EFS for shared model storage.  

---

## 📦 Model Weights

- Default: `models/best.pt` (YOLOv12m, 38 MB).  
- Options:
  - **Baked into image** (simple for deployment).  
  - **Mounted at runtime** via `docker-compose.yml`.  
  - **Downloaded at startup** from S3 (scales better).  

Set with env var:
```bash
MODEL_PATH=/app/models/best.pt
```

---

## 🧑‍💻 Contributors

- **Ronit Shahu** — End-to-end architecture, Dockerization, FastAPI service, Streamlit UI, MLflow integration.  
- Special thanks to open-source YOLOv12 community and MLflow maintainers.  

---

## 📜 License
MIT License. See [LICENSE](LICENSE) for details.
