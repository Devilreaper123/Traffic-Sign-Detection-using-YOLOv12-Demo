import os, queue, threading, time
from typing import Dict, Any
import mlflow

_MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
_EXP = os.getenv("MLFLOW_EXPERIMENT_NAME", "default")

_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=10000)
_started = False

def _worker():
    if not _MLFLOW_URI:  # tracking disabled
        return
    mlflow.set_tracking_uri(_MLFLOW_URI)
    mlflow.set_experiment(_EXP)
    while True:
        item = _q.get()
        if item is None:
            break
        try:
            with mlflow.start_run(run_name=item.get("run_name"), nested=True):
                # metrics
                for k, v in item.get("metrics", {}).items():
                    mlflow.log_metric(k, v)
                # params (logged once per run_name usually)
                for k, v in item.get("params", {}).items():
                    mlflow.log_param(k, v)
        except Exception:
            pass  # never block API
        finally:
            _q.task_done()

def start():
    global _started
    if _started: return
    _started = True
    t = threading.Thread(target=_worker, daemon=True)
    t.start()

def log_async(run_name: str, metrics: Dict[str, float], params: Dict[str, str] = None):
    if not _MLFLOW_URI:
        return
    try:
        _q.put_nowait({"run_name": run_name, "metrics": metrics, "params": params or {}})
    except queue.Full:
        pass
