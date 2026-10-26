"""Lazy, cached loading of Hugging Face pipelines. Torch/transformers are imported only when needed."""
from functools import lru_cache

class ModelUnavailable(RuntimeError):
    """Model could not be loaded; callers must show an 'unavailable' state, never fake output."""

@lru_cache(maxsize=4)
def load_pipeline(task: str, model_id: str):
    try:
        from transformers import pipeline
        return pipeline(task, model=model_id)
    except Exception as e:  # network, missing deps, bad id, out of memory
        raise ModelUnavailable(f"Could not load '{model_id}': {e}. See README > Models.") from e
