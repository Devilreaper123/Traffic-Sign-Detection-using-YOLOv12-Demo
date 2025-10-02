from pydantic import BaseModel, Field
from typing import List

class Box(BaseModel):
    cls: str = Field(..., description="Class name")
    conf: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    x1: int; y1: int; x2: int; y2: int

class Prediction(BaseModel):
    boxes: List[Box]
    latency_ms: float

class Health(BaseModel):
    status: str

class Warmup(BaseModel):
    ok: bool
    msg: str
