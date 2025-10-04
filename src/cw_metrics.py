# src/cw_metrics.py
import os, time, statistics
import boto3
from collections import deque
from threading import Lock

NAMESPACE = os.getenv("CW_NAMESPACE", "SignScopes/API")
DIM_SERVICE = os.getenv("CW_SERVICE", "yolo-api")
REGION = os.getenv("AWS_REGION", "us-east-2")

cw = boto3.client("cloudwatch", region_name=REGION)

class RollingLatency:
    def __init__(self, maxlen=500):
        self.q = deque(maxlen=maxlen)
        self.lock = Lock()

    def add(self, ms: float):
        with self.lock:
            self.q.append(ms)

    def snapshot(self):
        with self.lock:
            data = list(self.q)
        if not data:
            return None
        p50 = statistics.quantiles(data, n=100)[49] if len(data) >= 100 else statistics.median(data)
        p95 = sorted(data)[int(0.95*len(data))-1]
        return {"count": len(data), "p50": p50, "p95": p95}

lat_rolling = RollingLatency()

def put_metrics(p50_ms: float, p95_ms: float, tput_rps: float):
    dims = [{"Name": "Service", "Value": DIM_SERVICE}]
    ts = int(time.time())
    cw.put_metric_data(
        Namespace=NAMESPACE,
        MetricData=[
            {"MetricName": "LatencyP50", "Dimensions": dims, "Timestamp": ts, "Unit": "Milliseconds", "Value": p50_ms},
            {"MetricName": "LatencyP95", "Dimensions": dims, "Timestamp": ts, "Unit": "Milliseconds", "Value": p95_ms},
            {"MetricName": "ThroughputRPS", "Dimensions": dims, "Timestamp": ts, "Unit": "Count/Second", "Value": tput_rps},
        ],
    )
