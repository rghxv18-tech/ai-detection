import streamlit as st
from src import config as C
from src.validators import ValidationError
from src.utils.model_loader import ModelUnavailable
from src.utils.video_processing import VideoError
from src.detectors.text_detector import detect_text
from src.detectors.image_detector import detect_image
from src.detectors.video_detector import detect_video

st.set_page_config(page_title="AI Content & Deepfake Detector", page_icon="🔍", layout="wide")
st.title("🔍 AI Content & Deepfake Detector")
st.caption("Local, privacy-first analysis of text, images and video. Nothing you upload is stored.")
st.warning("Results are **probabilistic model scores, not proof** of authorship or authenticity. "
           "Detectors make false positives and false negatives. Never use them alone to accuse anyone.")

def run(fn, *a):
    try:
        with st.spinner("Analysing..."):
            return fn(*a)
    except (ValidationError, VideoError) as e:
        st.error(str(e))
    except ModelUnavailable as e:
        st.error(f"Model unavailable: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

t1, t2, t3 = st.tabs(["Text Detector", "Image Detector", "Video Detector"])

with t1:
    txt = st.text_area("Paste text", height=200)
    f = st.file_uploader("...or upload a .txt file", type=["txt"], key="t")
    if f:
        txt = f.read().decode("utf-8", errors="replace")
    if st.button("Analyze text"):
        r = run(detect_text, txt)
        if r:
            st.metric("Model label", r.label, f"score {r.score:.2f}")
            st.info("The default model targets GPT-2-era text and is outdated for modern LLMs. "
                    "Non-native English writers are often falsely flagged.")
            with st.expander("Technical details"):
                st.json({"model": r.model_id, "note": "score is a raw model output, not a calibrated probability"})

with t2:
    f = st.file_uploader("Upload image", type=[e[1:] for e in C.IMAGE_EXTS], key="i")
    if f and st.button("Analyze image"):
        r = run(detect_image, f.name, f.getvalue())
        if r:
            st.image(f.getvalue(), width=400)
            st.metric("Top label", r.top_label, f"{r.scores[r.top_label]:.2f}")
            with st.expander("Technical details"):
                st.json({"model": r.model_id, "scores": r.scores})

with t3:
    interval = st.slider("Sample one frame every N seconds", 0.5, 10.0, C.DEFAULT_FRAME_INTERVAL_S, 0.5)
    f = st.file_uploader("Upload video", type=[e[1:] for e in C.VIDEO_EXTS], key="v")
    if f and st.button("Analyze video"):
        r = run(detect_video, f.name, f.getvalue(), interval)
        if r:
            s = r.summary
            c = st.columns(4)
            c[0].metric("Mean fake score", f"{s['mean']:.2f}")
            c[1].metric("Median", f"{s['median']:.2f}")
            c[2].metric("Max", f"{s['max']:.2f}")
            c[3].metric("Frames flagged", f"{s['frac_flagged']:.0%}")
            st.caption(f"Method: {r.method}. Only {r.faces_found} sampled frames contained a detectable face; "
                       "this does not prove the whole video is authentic or fake.")
            for ts, img, sc in r.frames:
                st.image(img, caption=f"t={ts:.1f}s  fake score {sc:.2f}", width=160)
