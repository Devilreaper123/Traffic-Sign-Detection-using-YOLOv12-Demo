from pathlib import Path

# Path to your trained weights
WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "models" / "best.pt"

# Input size used in training
INPUT_SIZE = 320

# Class names (10 classes, order must match training)
CLASS_NAMES = [
    "Speed Limit 50",
    "Speed Limit 100",
    "No Overtaking",
    "Yield",
    "Stop",
    "No Entry",
    "Danger Ahead",
    "Road Work Ahead",
    "Pedestrian Crossing",
    "Children Crossing",
]
