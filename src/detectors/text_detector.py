from dataclasses import dataclass
from src import config as C
from src.validators import validate_text
from src.utils.model_loader import load_pipeline

@dataclass
class TextResult:
    label: str          # raw model label
    score: float        # raw model score for that label (NOT a calibrated probability)
    model_id: str

def preprocess(text: str) -> str:
    return " ".join(text.split())

def detect_text(text: str, pipe=None) -> TextResult:
    text = preprocess(validate_text(text))
    pipe = pipe or load_pipeline("text-classification", C.TEXT_MODEL_ID)
    out = pipe(text, truncation=True, max_length=512)[0]
    return TextResult(label=str(out["label"]), score=float(out["score"]), model_id=C.TEXT_MODEL_ID)
