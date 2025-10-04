# 🚦 Traffic Sign Detection using YOLOv12m

End-to-end **Dockerized YOLOv12m pipeline** for real-time traffic sign detection, benchmarking, and experiment tracking.  
Includes **FastAPI inference API**, **Streamlit benchmarking UI**, **MLflow tracking**, and **Caddy reverse proxy**, orchestrated via **Docker Compose** (CPU/GPU ready).  
Deployed on **AWS EC2 (c7i.xlarge)** with **EventBridge auto-scheduling** for 9 AM–9 PM uptime and cost efficiency.

---

## 🌐 Live Demo

| Service | URL | Description |
|----------|-----|-------------|
| 🧠 API | [https://api.signscopes.com/docs](https://api.signscopes.com/docs) | FastAPI inference endpoints |
| 🎛️ UI | [https://demo.signscopes.com](https://demo.signscopes.com) | Streamlit benchmark & visualization dashboard |
| 📊 MLflow | [https://mlflow.signscopes.com](https://mlflow.signscopes.com) | Experiment tracking and latency analytics |

---

## 📌 Architecture

```
                 ┌────────────────────────────┐
                 │        Streamlit UI        │
                 │ (demo.signscopes.com)      │
                 └─────────────▲──────────────┘
                               │ REST calls
                               │
                 ┌─────────────┴──────────────┐
                 │        FastAPI API         │
                 │ (api.signscopes.com)       │
                 └─────────────▲──────────────┘
                               │ metrics + logs
                               │
                 ┌─────────────┴──────────────┐
                 │           MLflow           │
                 │ (mlflow.signscopes.com)    │
                 └─────────────▲──────────────┘
                               │ reverse proxy
                               │
                 ┌─────────────┴──────────────┐
                 │        Caddy Proxy         │
                 │ SSL + routing              │
                 └────────────────────────────┘
```

---

## ✅ Features

- 🚀 **FastAPI** serving YOLOv12m via `/predict` and `/predict_batch`
- 🎨 **Streamlit** UI for benchmarking latency & throughput
- 📊 **MLflow** integrated for tracking inference performance
- 🐳 **Docker Compose** orchestration (API + UI + MLflow + Caddy)
- ⚙️ **Caddy Reverse Proxy** for HTTPS & multi-service routing
- 🧠 **AWS EC2 (c7i.xlarge)** with **EventBridge auto start/stop**
- 🔍 **Prometheus-compatible metrics** exposed at `/metrics`
- 🔐 Environment-based configuration for flexible deployment

---

## ⚡ Benchmarks (YOLOv12m @ imgsz=320)

| Mode | Batch Size | Total Time | Throughput | Server p50 | p95 | End-to-End |
|------|-------------|-------------|-------------|-------------|-------------|-------------|
| 🖼️ Single | 1 | – | – | **523 ms** | **523 ms** | **580 ms** |
| 🔁 Sequential | 30 | **5.30 s** | **5.66 img/s** | **130 ms** | **145 ms** | – |
| 📦 Batch | 30 | **3.88 s** | **7.74 img/s** | **130 ms** | **141 ms** | – |
| ✅ Success Rate | – | – | – | **100 %** | **0 Errors** | – |

> Deployed via Dockerized FastAPI + Streamlit stack on AWS EC2 (c7i.xlarge) with 4 Uvicorn workers.  
> Auto-start/stop handled by **EventBridge Scheduler** to minimize cost and idle runtime.

---

## 🛠️ Setup (Local or EC2)

### 1️⃣ Clone the repo
```bash
git clone https://github.com/Devilreaper123/Traffic-Sign-Detection-using-YOLOv12-Demo.git
cd Traffic-Sign-Detection-using-YOLOv12-Demo
```

### 2️⃣ Build and run
```bash
docker compose up -d --build
```

### 3️⃣ Access services locally
- API → [http://localhost:8000/docs](http://localhost:8000/docs)  
- UI → [http://localhost:8501](http://localhost:8501)  
- MLflow → [http://localhost:5000](http://localhost:5000)

---

## 🧰 Docker Quick Commands

```bash
docker ps -a               # list containers
docker logs yolo-api       # view logs
docker exec -it yolo-api sh
docker compose down        # stop services
docker system prune -af    # clean up
```

---

## 🧾 Environment Variables

| Key | Example | Purpose |
|-----|----------|----------|
| `MODEL_PATH` | `/app/models/best.pt` | YOLO weights path |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow backend |
| `MLFLOW_EXPERIMENT_NAME` | `yolov12m-traffic-sign` | Experiment name |
| `API_URL` | `https://api.signscopes.com` | Base URL for Streamlit |
| `UVICORN_WORKERS` | `4` | API concurrency tuning |

---

## 🚀 CI/CD with GitHub Actions

Automated Docker build + push to DockerHub on every `main` push.

```yaml
- uses: docker/build-push-action@v6
  with:
    context: .
    file: Dockerfile
    push: true
    tags: |
      ronitshahu/traffic-sign-yolo:latest
      ronitshahu/traffic-sign-ui:latest
```

---

## ☁️ AWS Deployment Overview

- **Instance**: EC2 `c7i.xlarge` (4 vCPUs, 8 GB RAM)  
- **Reverse Proxy**: Caddy (SSL via HTTPS for all services)  
- **Auto Scheduler**: AWS **EventBridge**
  - Start → 9 AM (EST)
  - Stop → 9 PM (EST)
- **Data Retention**: MLflow logs persisted on volume `/mlruns`
- **Domains**:  
  - `api.signscopes.com` → FastAPI  
  - `demo.signscopes.com` → Streamlit  
  - `mlflow.signscopes.com` → MLflow  

---

## 📈 MLflow Tracking

Automatically logs:
- Inference latency (`latency_ms`, `batch_latency_ms`)
- Throughput, box counts, and class-wise metrics
- Run metadata (commit SHA, model version)

MLflow UI: [https://mlflow.signscopes.com](https://mlflow.signscopes.com)

---

## 🧑‍💻 Contributors

- **Ronit Shahu** — Architecture, FastAPI, Streamlit UI, Dockerization, MLflow, CI/CD, AWS deployment.  
- Thanks to the **Ultralytics YOLOv12** and **MLflow** open-source community.

---

## 📜 License

MIT License — see [LICENSE](LICENSE).
