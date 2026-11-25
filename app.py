import os
import tempfile
import time
import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import APIError, ServerError
from PIL import ExifTags, Image
from pydantic import BaseModel, Field

# 1. Setup Page & API Key
st.set_page_config(
    page_title="VeriAI - Scam & Deepfake Detector",
    page_icon="🛡️",
    layout="centered",
)
st.title("🛡️ VeriAI: Multi-Modal Fake & Scam Detector")
st.caption(
    "Detect phishing messages, manipulated images, cloned voices, and deepfake videos."
)


def get_initial_api_key() -> str:
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
        if secret_key:
            return str(secret_key).strip()
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "").strip()


default_api_key = get_initial_api_key()

with st.sidebar:
    api_key = st.text_input(
        "Enter Gemini API Key",
        value=default_api_key,
        type="password",
    ).strip()

if not api_key:
    st.info("👈 Paste your Gemini API Key in the sidebar to start.")
    st.stop()

client = genai.Client(api_key=api_key)

# Models tried in order if one hits a temporary 503 traffic spike
MODELS_TO_TRY = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-4-hydrogen",
]


# 2. Structured Output Schema
class ForensicsReport(BaseModel):
    verdict: str = Field(
        description="One of: LIKELY FAKE / SCAM, SUSPICIOUS, or LIKELY AUTHENTIC"
    )
    risk_score: int = Field(
        description="Risk score from 0 (completely safe/real) to 100 (definitely fake/scam)"
    )
    red_flags: list[str] = Field(
        description="3 to 5 specific visual, metadata, audio, or linguistic clues found"
    )
    explanation: str = Field(
        description="Clear, simple explanation for a non-technical user"
    )
    recommended_action: str = Field(
        description="What the user should do right now"
    )


SYSTEM_PROMPT = """
You are an expert digital forensics and cybersecurity analyst. Analyze the provided content 
(text, email, image + EXIF metadata, audio, or video) for signs of fraud, phishing, manipulation, or AI generation.
Look for:
- Text/Emails: Urgency tactics, fake domains, grammatical anomalies, suspicious payment/OTP requests.
- Images & Metadata: Warped hands/text, inconsistent shadows/lighting, mismatched fonts in screenshots, AND missing camera EXIF data or editing software tags (e.g., Adobe Photoshop, Midjourney). Note: WhatsApp/social media compression strips EXIF, so weigh visual evidence alongside metadata.
- Audio: Unnatural breathing, flat emotional cadence, metallic artifacts, scam scripts (e.g., fake police/bank calls).
- Video: Lip-sync delays, unnatural blinking, blurring around the jawline/hair, inconsistent background physics.
Be objective, highlight exact evidence, and never exaggerate certainty.
"""


def extract_exif_data(img: Image.Image) -> dict:
    """Extracts human-readable camera EXIF metadata from a PIL Image."""
    exif_dict = {}
    raw_exif = img.getexif()
    if not raw_exif:
        return exif_dict

    important_tags = {
        "Make",
        "Model",
        "Software",
        "DateTime",
        "DateTimeOriginal",
        "ExposureTime",
        "FNumber",
        "ISOSpeedRatings",
        "FocalLength",
        "LensModel",
    }

    for tag_id, value in raw_exif.items():
        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
        if tag_name in important_tags:
            if isinstance(value, bytes):
                value = value.decode(errors="ignore").strip("\x00")
            exif_dict[tag_name] = str(value)

    try:
        exif_ifd = raw_exif.get_ifd(ExifTags.IFD.Exif)
        for tag_id, value in exif_ifd.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
            if tag_name in important_tags:
                if isinstance(value, bytes):
                    value = value.decode(errors="ignore").strip("\x00")
                exif_dict[tag_name] = str(value)
    except Exception:
        pass

    return exif_dict


def analyze_content(contents) -> ForensicsReport | None:
    with st.spinner("Running AI forensic analysis..."):
        last_error = None
        for model_name in MODELS_TO_TRY:
            for attempt in range(2):
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            response_mime_type="application/json",
                            response_schema=ForensicsReport,
                            temperature=0.2,
                        ),
                    )
                    return ForensicsReport.model_validate_json(response.text)
                except ServerError as e:
                    last_error = e
                    time.sleep(2 * (attempt + 1))
                except APIError as e:
                    if e.code == 503:
                        last_error = e
                        time.sleep(2 * (attempt + 1))
                    else:
                        st.error(f"Gemini API error ({model_name}): {e.message}")
                        return None

        st.error(
            f"All models are currently experiencing high traffic ({last_error}). Please wait 10–15 seconds and try again."
        )
        return None


def display_report(report: ForensicsReport):
    if report.risk_score >= 70:
        st.error(
            f"### Verdict: {report.verdict} (Risk Score: {report.risk_score}/100)"
        )
    elif report.risk_score >= 35:
        st.warning(
            f"### Verdict: {report.verdict} (Risk Score: {report.risk_score}/100)"
        )
    else:
        st.success(
            f"### Verdict: {report.verdict} (Risk Score: {report.risk_score}/100)"
        )

    st.progress(report.risk_score / 100)
    st.markdown(f"**Summary:** {report.explanation}")

    st.markdown("**🔍 Key Indicators Found:**")
    for flag in report.red_flags:
        st.markdown(f"- {flag}")

    st.info(f"**💡 Recommended Action:** {report.recommended_action}")


def upload_temp_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        gemini_file = client.files.upload(file=tmp_path)
        while gemini_file.state.name == "PROCESSING":
            time.sleep(2)
            gemini_file = client.files.get(name=gemini_file.name)
        return gemini_file
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# 3. Four Tabs for Your Hackathon Demo
tab1, tab2, tab3, tab4 = st.tabs([
    "📧 Email / Message",
    "🖼️ Image / Screenshot",
    "🎙️ Voice / Audio",
    "🎥 Video",
])

with tab1:
    msg = st.text_area(
        "Paste suspicious email, SMS, or WhatsApp message:", height=150
    )
    if st.button("Analyze Text", key="btn_text") and msg:
        report = analyze_content([
            "Analyze this message for phishing, scam, or AI-generated fraud:",
            msg,
        ])
        if report:
            display_report(report)

with tab2:
    img_file = st.file_uploader(
        "Upload photo or payment screenshot", type=["jpg", "jpeg", "png"]
    )
    if img_file:
        img = Image.open(img_file)
        exif_data = extract_exif_data(img)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(img, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.markdown("#### 📷 Camera EXIF Forensics")
            if exif_data:
                st.success("✅ Hardware/Software EXIF Metadata Found")
                st.table([{"Tag": k, "Value": v} for k, v in exif_data.items()])
            else:
                st.warning("⚠️ No Camera EXIF Metadata Found")
                st.caption(
                    "Common in AI-generated images, screenshots, or photos stripped by messaging apps (like WhatsApp)."
                )

        if st.button("Analyze Image & Metadata", key="btn_img"):
            exif_summary = (
                str(exif_data)
                if exif_data
                else "None (Metadata stripped or never existed)"
            )
            prompt = (
                "Inspect this image for AI artifacts, photoshopping, or fake screenshot signs.\n"
                f"Extracted EXIF Metadata: {exif_summary}"
            )
            report = analyze_content([prompt, img])
            if report:
                display_report(report)

with tab3:
    audio_file = st.file_uploader(
        "Upload voice note or call recording", type=["mp3", "wav", "m4a", "ogg"]
    )
    if audio_file:
        st.audio(audio_file)
        if st.button("Analyze Audio", key="btn_audio"):
            uploaded = upload_temp_file(audio_file)
            report = analyze_content([
                uploaded,
                "Analyze this audio for AI voice cloning artifacts and scam call tactics.",
            ])
            if report:
                display_report(report)

with tab4:
    video_file = st.file_uploader(
        "Upload short video clip (under 20MB for fast demo)",
        type=["mp4", "mov", "webm"],
    )
    if video_file:
        st.video(video_file)
        if st.button("Analyze Video", key="btn_video"):
            uploaded = upload_temp_file(video_file)
            report = analyze_content([
                uploaded,
                "Analyze this video for deepfake lip-sync issues, facial warping, or synthetic generation.",
            ])
            if report:
                display_report(report)