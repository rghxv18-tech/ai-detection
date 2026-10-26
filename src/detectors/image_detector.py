from dataclasses import dataclass
from src import config as C
from src.validators import validate_image_bytes
from src.utils.model_loader import load_pipeline

@dataclass
class ImageResult:
    scores: dict        # label -> model score
    top_label: str
    model_id: str

def classify_image(img, model_id: str, pipe=None) -> dict:
    pipe = pipe or load_pipeline("image-classification", model_id)
    return {r["label"]: float(r["score"]) for r in pipe(img)}

def detect_image(name: str, data: bytes, pipe=None) -> ImageResult:
    img = validate_image_bytes(name, data)
    scores = classify_image(img, C.IMAGE_MODEL_ID, pipe)
    return ImageResult(scores, max(scores, key=scores.get), C.IMAGE_MODEL_ID)
