import os
TEXT_MODEL_ID = os.getenv("TEXT_MODEL_ID", "openai-community/roberta-base-openai-detector")
IMAGE_MODEL_ID = os.getenv("IMAGE_MODEL_ID", "umm-maybe/AI-image-detector")
DEEPFAKE_MODEL_ID = os.getenv("DEEPFAKE_MODEL_ID", "dima806/deepfake_vs_real_image_detection")
MAX_TEXT_CHARS = 20_000
MIN_TEXT_CHARS = 50
MAX_IMAGE_MB = 10
MAX_IMAGE_PIXELS = 4096 * 4096
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_VIDEO_MB = 100
MAX_VIDEO_SECONDS = 60
VIDEO_EXTS = {".mp4", ".mov"}
DEFAULT_FRAME_INTERVAL_S = 2.0
MAX_FRAMES = 30
