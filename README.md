# AI Content & Deepfake Detector

Streamlit app that estimates whether text, images or videos may be AI-generated or manipulated. **All outputs are probabilistic model scores, not proof.**

*Screenshots: add `docs/text.png`, `docs/image.png`, `docs/video.png` here.*

## Setup (Windows, PowerShell) - Python 3.13.11
`.python-version` and `pyproject.toml` pin **3.13.11** (the newest 3.13 release verified when this was written; edit both files to change it).
```powershell
py -3.13 --version                 # must print Python 3.13.11
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt    # verify torch/opencv wheels exist for 3.13 on your OS
pytest
streamlit run app.py
```
After a successful install: `pip freeze > requirements.lock.txt` (the lockfile does not exist yet).

## Models (configure in `.env`, copy from `.env.example`)
Models download from Hugging Face on first use (hundreds of MB each) and run locally. **Verify each model card's license, size and intended use before relying on or redistributing them; licenses below are not verified.**

| Task | Default ID | Notes |
|---|---|---|
| Text | `openai-community/roberta-base-openai-detector` | Trained on GPT-2 output; weak on modern LLMs; high false-positive risk |
| Image | `umm-maybe/AI-image-detector` | Trained on limited generators; accuracy drops on new ones |
| Video (face frames) | `dima806/deepfake_vs_real_image_detection` | Image-level face classifier applied per frame; label names must contain "fake" |

If a model fails to load, the app shows an "unavailable" error. It never fabricates predictions.

## Video method
Frames sampled every N seconds (max 30, max 60 s, 100 MB) -> OpenCV Haar face crop (basic; misses small/angled faces) -> per-frame fake score -> mean / median / max / fraction >0.5. Faceless AI video is not covered.

## Evaluating models
Use labelled data you are licensed to use (e.g. a held-out human vs AI text set; real vs generated image sets such as CIFAKE; FaceForensics++ / Celeb-DF for video, which require access agreements). Metrics: **precision** = TP/(TP+FP); **recall** = TP/(TP+FN); **F1** = harmonic mean of both; **confusion matrix** = counts of TP/FP/FN/TN; **false-positive rate** = FP/(FP+TN), i.e. real content wrongly flagged. No evaluation tooling is implemented yet.

## Tests
`tests/` use mocked models only; they check validation, frame sampling, aggregation and error handling, not detector accuracy.

## Privacy / security
Local inference, uploads processed in memory or temp files deleted after use, nothing stored. Do not commit `models/`, `data/` or `.env`.

## Troubleshooting & deployment
Torch install fails -> check Python 3.13 wheels exist; try CPU build from pytorch.org. Slow first run -> model download. Deploy on a host with enough RAM (>=4 GB); set model IDs via env vars.

## Future work
Calibrated text model trained on your own data, evaluation scripts, better face detector, audio-visual checks, C2PA provenance.
