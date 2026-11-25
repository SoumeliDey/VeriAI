import os
import tempfile
import time
from typing import List, Literal

import streamlit as st
from PIL import ExifTags, Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="VeriAI - Scam & Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
)

# Models tried in order if one hits a 503 / high-traffic spike
MODELS_TO_TRY = [
    MODELS_TO_TRY = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-3.1-pro-preview",
]
]


# ---------------------------------------------------------
# 2. Structured Forensic Report Schema
# ---------------------------------------------------------
class ForensicsReport(BaseModel):
    verdict: Literal["Likely Authentic", "Suspicious", "High Risk / Scam / Deepfake"] = Field(
        description="Overall authenticity and safety verdict."
    )
    risk_score: int = Field(
        ge=0,
        le=100,
        description="Risk score from 0 (completely safe/authentic) to 100 (confirmed scam/deepfake).",
    )
    red_flags: List[str] = Field(
        description="Specific forensic anomalies, manipulation clues, or social-engineering red flags."
    )
    explanation: str = Field(
        description="Clear, concise summary explaining why this verdict and risk score were assigned."
    )
    recommended_action: str = Field(
        description="Concrete next step the user should take to stay safe."
    )


# ---------------------------------------------------------
# 3. Secure Server-Side API Key Resolution
# ---------------------------------------------------------
def get_server_api_key() -> str:
    """Read API key securely on the server without ever exposing it to the client UI."""
    try:
        if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"]:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "").strip()


server_api_key = get_server_api_key()

with st.sidebar:
    st.header("⚙️ Configuration")
    if server_api_key:
        api_key = server_api_key
        st.success("✅ API key configured securely on server")
    else:
        api_key = st.text_input(
            "Enter Gemini API Key",
            type="password",
            placeholder="AIzaSy...",
            help="Stored only in your current browser session.",
        )

    st.markdown("---")
    st.markdown(
        "**How VeriAI Works:**\n"
        "- **Text:** Detects urgency traps, phishing links & impersonation.\n"
        "- **Image:** Combines local EXIF hardware forensics with visual artifact checks.\n"
        "- **Audio/Video:** Analyzes voice cadence, lip-sync & synthetic artifacts."
    )


# ---------------------------------------------------------
# 4. Helper Functions (EXIF, Temp Upload, AI Analysis)
# ---------------------------------------------------------
def extract_exif_data(image: Image.Image) -> dict:
    """Extract readable EXIF camera/software tags from a PIL Image."""
    exif_summary = {}
    try:
        raw_exif = image.getexif()
        if not raw_exif:
            return exif_summary

        interesting_tags = {
            "Make",
            "Model",
            "Software",
            "DateTime",
            "DateTimeOriginal",
            "ExposureTime",
            "FNumber",
            "ISOSpeedRatings",
            "FocalLength",
        }
        for tag_id, value in raw_exif.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
            if tag_name in interesting_tags:
                exif_summary[tag_name] = str(value)
    except Exception:
        pass
    return exif_summary


def upload_temp_file(client: genai.Client, uploaded_file):
    """Save uploaded media briefly to a temp file, upload to Gemini Files API, and delete locally."""
    suffix = os.path.splitext(uploaded_file.name)[1] or ".bin"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        gemini_file = client.files.upload(file=tmp_path)
        while getattr(gemini_file, "state", None) and gemini_file.state.name == "PROCESSING":
            time.sleep(2)
            gemini_file = client.files.get(name=gemini_file.name)

        return gemini_file
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def analyze_content(api_key: str, contents, system_instruction: str) -> ForensicsReport:
    """Run Gemini forensic analysis with automatic retry and model fallback."""
    client = genai.Client(api_key=api_key)
    last_error = None

    for model_name in MODELS_TO_TRY:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=ForensicsReport,
                        temperature=0.2,
                    ),
                )
                return ForensicsReport.model_validate_json(response.text)
            except Exception as exc:
                last_error = exc
                err_str = str(exc)
                if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str:
                    time.sleep(2 * (attempt + 1))
                    continue
                break

    raise RuntimeError(f"All forensic models are currently busy or returned an error: {last_error}")


def display_report(report: ForensicsReport):
    """Render structured forensic results cleanly."""
    score = report.risk_score
    if score >= 70 or "High Risk" in report.verdict:
        st.error(f"### 🚨 Verdict: {report.verdict} (Risk Score: {score}/100)")
    elif score >= 35 or report.verdict == "Suspicious":
        st.warning(f"### ⚠️ Verdict: {report.verdict} (Risk Score: {score}/100)")
    else:
        st.success(f"### ✅ Verdict: {report.verdict} (Risk Score: {score}/100)")

    st.progress(min(max(score, 0), 100) / 100.0)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("#### 🔍 Forensic Explanation")
        st.write(report.explanation)
        st.markdown("#### 🛡️ Recommended Action")
        st.info(report.recommended_action)

    with col2:
        st.markdown("#### 🚩 Key Red Flags & Signals")
        if report.red_flags:
            for flag in report.red_flags:
                st.markdown(f"- {flag}")
        else:
            st.markdown("- No major red flags detected.")


# ---------------------------------------------------------
# 5. Main User Interface
# ---------------------------------------------------------
st.title("🛡️ VeriAI: Multi-Modal Fake & Scam Detector")
st.caption(
    "Detect phishing messages, manipulated screenshots, AI-generated images, and audio/video deepfakes."
)

tab_text, tab_image, tab_audio, tab_video = st.tabs(
    ["✉️ Email / Message", "🖼️ Image / Screenshot", "🎙️ Voice / Audio", "🎬 Video"]
)

# --- TAB 1: EMAIL / MESSAGE ---
with tab_text:
    st.subheader("Analyze Suspicious Emails, SMS, or Chat Messages")
    text_input = st.text_area(
        "Paste the message content, sender details, or suspicious URL below:",
        height=180,
        placeholder="Example: URGENT: Your bank account has been suspended. Click http://verify-kyc-now.example to restore access within 2 hours...",
    )
    if st.button("Analyze Text", type="primary", key="btn_text"):
        if not api_key:
            st.error("Please configure GEMINI_API_KEY in Streamlit Secrets or enter a key in the sidebar.")
        elif not text_input.strip():
            st.warning("Please paste a message or email to analyze.")
        else:
            with st.spinner("Running AI scam & social-engineering analysis..."):
                try:
                    system_prompt = (
                        "You are an expert cybersecurity and anti-fraud analyst. Evaluate the provided message "
                        "for phishing, social engineering, urgency manipulation, spoofed domains, payment scams, "
                        "and impersonation."
                    )
                    report = analyze_content(api_key, text_input, system_prompt)
                    display_report(report)
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

# --- TAB 2: IMAGE / SCREENSHOT ---
with tab_image:
    st.subheader("Analyze Photos, Payment Screenshots, or Suspected AI Images")
    uploaded_img = st.file_uploader(
        "Upload an image (JPG, JPEG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        key="up_img",
    )
    if uploaded_img is not None:
        pil_img = Image.open(uploaded_img)
        col_preview, col_meta = st.columns([1, 1])

        with col_preview:
            st.image(pil_img, caption=uploaded_img.name, use_container_width=True)

        with col_meta:
            st.markdown("#### 📋 Local EXIF Metadata Forensics")
            exif_info = extract_exif_data(pil_img)
            if exif_info:
                st.table([{"Tag": k, "Value": v} for k, v in exif_info.items()])
            else:
                st.info(
                    "No camera EXIF metadata found (common in screenshots, social media downloads, or AI-generated images)."
                )

        if st.button("Analyze Image & Metadata", type="primary", key="btn_img"):
            if not api_key:
                st.error("Please configure GEMINI_API_KEY in Streamlit Secrets or enter a key in the sidebar.")
            else:
                with st.spinner("Running AI forensic analysis..."):
                    try:
                        exif_context = (
                            f"Extracted EXIF metadata: {exif_info}"
                            if exif_info
                            else "Extracted EXIF metadata: None found."
                        )
                        system_prompt = (
                            "You are a digital media forensics expert. Analyze both the image and its EXIF metadata "
                            "for signs of AI generation, Photoshop/font tampering (e.g., fake payment receipts), "
                            "anatomical/lighting inconsistencies, or scam indicators."
                        )
                        report = analyze_content(api_key, [pil_img, exif_context], system_prompt)
                        display_report(report)
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

# --- TAB 3: VOICE / AUDIO ---
with tab_audio:
    st.subheader("Analyze Voice Notes or Phone Call Recordings")
    uploaded_audio = st.file_uploader(
        "Upload an audio file (MP3, WAV, M4A, OGG)",
        type=["mp3", "wav", "m4a", "ogg"],
        key="up_audio",
    )
    if uploaded_audio is not None:
        st.audio(uploaded_audio)
        if st.button("Analyze Audio", type="primary", key="btn_audio"):
            if not api_key:
                st.error("Please configure GEMINI_API_KEY in Streamlit Secrets or enter a key in the sidebar.")
            else:
                with st.spinner("Uploading and analyzing audio for voice cloning & scam cues..."):
                    try:
                        client = genai.Client(api_key=api_key)
                        gemini_audio = upload_temp_file(client, uploaded_audio)
                        system_prompt = (
                            "You are an audio forensics and fraud specialist. Analyze this audio clip for "
                            "synthetic voice cloning artifacts (flat prosody, unnatural breathing, metallic splices) "
                            "as well as vishing / impersonation scam tactics."
                        )
                        report = analyze_content(
                            api_key,
                            [gemini_audio, "Analyze this audio recording for deepfake voice cloning or scam intent."],
                            system_prompt,
                        )
                        display_report(report)
                    except Exception as e:
                        st.error(f"Audio analysis failed: {e}")

# --- TAB 4: VIDEO ---
with tab_video:
    st.subheader("Analyze Video Clips for Deepfakes or Manipulated Media")
    uploaded_video = st.file_uploader(
        "Upload a short video clip (MP4, MOV, WEBM)",
        type=["mp4", "mov", "webm"],
        key="up_video",
    )
    if uploaded_video is not None:
        st.video(uploaded_video)
        if st.button("Analyze Video", type="primary", key="btn_video"):
            if not api_key:
                st.error("Please configure GEMINI_API_KEY in Streamlit Secrets or enter a key in the sidebar.")
            else:
                with st.spinner("Uploading and analyzing video frames & audio track..."):
                    try:
                        client = genai.Client(api_key=api_key)
                        gemini_video = upload_temp_file(client, uploaded_video)
                        system_prompt = (
                            "You are a video deepfake forensics analyst. Inspect facial boundaries, blinking, "
                            "lip-sync alignment, temporal flickering, lighting consistency, and spoken scam claims."
                        )
                        report = analyze_content(
                            api_key,
                            [gemini_video, "Analyze this video for deepfake manipulation or fraudulent claims."],
                            system_prompt,
                        )
                        display_report(report)
                    except Exception as e:
                        st.error(f"Video analysis failed: {e}")
