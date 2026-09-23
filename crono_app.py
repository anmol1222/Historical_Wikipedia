import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import torch
import soundfile as sf
import pandas as pd
import streamlit as st
from PIL import Image
from datasets import Dataset, load_from_disk

BASE_DIR = Path(__file__).resolve().parent
NLP_MODEL = os.getenv("CHRONODOC_NLP_MODEL", "google/flan-t5-small")
T2I_MODEL = os.getenv("CHRONODOC_T2I_MODEL", "segmind/tiny-sd")

if not torch.cuda.is_available():
    torch.set_num_threads(min(4, os.cpu_count() or 1))

# -------------------------------------------------------------
# 1. Page Configuration & Title
# -------------------------------------------------------------
st.set_page_config(page_title="ChronoDoc Multimodal AI", layout="wide")
st.title("ChronoDoc || Multimodal Historical Archivist ||")
st.write("An end-to-end multimodal pipeline linking STT, I2T, NLP, T2I, TTS, and S2S.")

if torch.cuda.is_available():
    device = "cuda"
    device_label = "CUDA GPU"
else:
    device = "cpu"
    device_label = "CPU"

st.caption(f"Compute device: {device_label}")


def get_pipeline():
    from transformers import pipeline

    return pipeline


def find_asset(*relative_paths):
    for relative_path in relative_paths:
        candidate = BASE_DIR / relative_path
        if candidate.exists():
            return candidate
    return None


@st.cache_resource
def load_dataset():
    dataset_path = BASE_DIR / "chrono_unified_data"
    if dataset_path.exists():
        return load_from_disk(str(dataset_path))

    csv_paths = {
        "audio": BASE_DIR / "audio_speech_5k.csv",
        "vision": BASE_DIR / "historical_image_text_5k.csv",
        "qa": BASE_DIR / "instruction_qa_5k.csv",
    }
    missing_files = [str(path.name) for path in csv_paths.values() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing dataset source files: " + ", ".join(missing_files)
        )

    audio = pd.read_csv(csv_paths["audio"])
    vision = pd.read_csv(csv_paths["vision"])
    qa = pd.read_csv(csv_paths["qa"])
    limit = min(70, len(audio), len(vision), len(qa))
    merged = {
        "id": [f"chrono_{index:04d}" for index in range(limit)],
        "audio_filename": audio["audio_file"].iloc[:limit].tolist(),
        "speaker_dialect": audio["dialect"].iloc[:limit].tolist(),
        "ground_transcript": audio["transcript"].iloc[:limit].tolist(),
        "image_id": vision["image_id"].iloc[:limit].tolist(),
        "artifact_title": vision["title"].iloc[:limit].tolist(),
        "historic_era": vision["era"].iloc[:limit].tolist(),
        "vision_caption": vision["caption"].iloc[:limit].tolist(),
        "user_instruction": qa["instruction"].iloc[:limit].tolist(),
        "historic_question": qa["question"].iloc[:limit].tolist(),
        "archival_context": qa["context"].iloc[:limit].tolist(),
        "expected_answer": qa["answer"].iloc[:limit].tolist(),
    }
    return Dataset.from_dict(merged)

# -------------------------------------------------------------
# 2. Cached Pipeline Loaders (Loads once to optimize memory)
# -------------------------------------------------------------
@st.cache_resource
def load_speech_to_text():
    return get_pipeline()("automatic-speech-recognition", model="openai/whisper-tiny", device=device)

@st.cache_resource
def load_image_to_text():
    return get_pipeline()("image-text-to-text", model="Salesforce/blip-image-captioning-base", device=device)

@st.cache_resource
def load_nlp_explainer():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(NLP_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(NLP_MODEL).to(device)
    model.eval()

    def explain(text, max_length=120):
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=False,
            )
        return [{"generated_text": tokenizer.decode(output_tokens[0], skip_special_tokens=True)}]

    return explain

@st.cache_resource
def load_text_to_speech():
    from transformers import AutoProcessor, VitsModel

    processor = AutoProcessor.from_pretrained("facebook/mms-tts-eng")
    model = VitsModel.from_pretrained("facebook/mms-tts-eng").to(device)
    model.eval()

    def synthesize(text):
        inputs = processor(text=text, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            waveform = model(**inputs).waveform
        return {
            "audio": waveform.cpu().numpy(),
            "sampling_rate": model.config.sampling_rate,
        }

    return synthesize

@st.cache_resource
def load_translation():
    from transformers import AutoModelForSeq2SeqLM, MarianTokenizer

    model_name = "Helsinki-NLP/opus-mt-en-hi"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    def translate(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output_tokens = model.generate(**inputs, max_new_tokens=128)
        translated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return [{"translation_text": translated_text}]

    return translate

@st.cache_resource
def load_hindi_tts():
    from transformers import AutoProcessor, VitsModel

    processor = AutoProcessor.from_pretrained("facebook/mms-tts-hin")
    model = VitsModel.from_pretrained("facebook/mms-tts-hin").to(device)
    model.eval()

    def synthesize(text):
        inputs = processor(text=text, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            waveform = model(**inputs).waveform
        return {
            "audio": waveform.cpu().numpy(),
            "sampling_rate": model.config.sampling_rate,
        }

    return synthesize

@st.cache_resource
def load_diffuser():
    from diffusers import StableDiffusionPipeline
    from diffusers import DDIMScheduler

    diffuser = StableDiffusionPipeline.from_pretrained(
        T2I_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    diffuser.scheduler = DDIMScheduler.from_config(diffuser.scheduler.config)
    diffuser.enable_attention_slicing()
    if hasattr(diffuser.vae, "enable_slicing"):
        diffuser.vae.enable_slicing()
    diffuser.set_progress_bar_config(disable=True)
    return diffuser.to(device)

# -------------------------------------------------------------
# 3. Sidebar: Select Artifact Record from Merged Dataset
# -------------------------------------------------------------
st.sidebar.header("Dataset Record Selection")

try:
    dataset = load_dataset()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()
record_ids = [f"{i}: {dataset[i]['artifact_title']}" for i in range(len(dataset))]
selected_idx = st.sidebar.selectbox("Choose an artifact row:", range(len(record_ids)), format_func=lambda x: record_ids[x])

sample = dataset[selected_idx]

st.sidebar.markdown(f"**Era:** {sample['historic_era']}")
st.sidebar.markdown(f"**Audio File:** `{sample['audio_filename']}`")
st.sidebar.markdown(f"**Image ID:** `{sample['image_id']}`")
quality_presets = {
    "Draft (fast)": {"steps": 10, "size": 256},
    "Standard": {"steps": 18, "size": 384},
    "High quality": {"steps": 28, "size": 512},
}
quality_name = st.sidebar.selectbox("Reconstruction quality:", list(quality_presets), index=1)
quality = quality_presets[quality_name]

# -------------------------------------------------------------
# 4. Main Interface: Input Inspection
# -------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Artifact Image")
    image_file = find_asset(
        f"{sample['image_id']}.jpg",
        f"images/{sample['image_id']}.jpg",
        f"assets/images/{sample['image_id']}.jpg",
    )
    if image_file is not None:
        input_image = Image.open(image_file)
        st.image(input_image, caption=sample['artifact_title'], use_container_width=True)
    else:
        preview_image = find_asset("reconstructed_artifact.png")
        input_image = None
        if preview_image is not None:
            st.image(preview_image, caption="Generated artifact preview", use_container_width=True)
        else:
            st.info("Original artifact image is unavailable; using the archival caption.")
        st.markdown(f"**Archival Caption:** {sample['vision_caption']}")

with col2:
    st.subheader("Input Spoken Audio Query")
    audio_file = find_asset(
        sample["audio_filename"],
        f"audio/{sample['audio_filename']}",
        f"assets/audio/{sample['audio_filename']}",
    )
    if audio_file is not None:
        st.audio(str(audio_file))
    else:
        preview_audio = find_asset("narration_output.wav")
        if preview_audio is not None:
            st.audio(str(preview_audio))
            st.caption("Generated narration preview; original query audio is unavailable.")
        else:
            st.info("Original query audio is unavailable; using the reference transcript.")
        st.markdown(f"**Reference Transcript:** {sample['ground_transcript']}")

st.markdown("---")

# -------------------------------------------------------------
# 5. Lazy Pipeline Execution
# -------------------------------------------------------------
st.header("Run Individual Archive Features")
st.caption("Each action loads only the models needed for that feature.")

if "transcription" not in st.session_state:
    st.session_state.transcription = None
if "detected_visuals" not in st.session_state:
    st.session_state.detected_visuals = None
if "explanation" not in st.session_state:
    st.session_state.explanation = None
if "hindi_text" not in st.session_state:
    st.session_state.hindi_text = None
if "narration_path" not in st.session_state:
    st.session_state.narration_path = None
if "hindi_audio_path" not in st.session_state:
    st.session_state.hindi_audio_path = None
if "reconstructed_image_path" not in st.session_state:
    st.session_state.reconstructed_image_path = None

action_col1, action_col2, action_col3 = st.columns(3)
with action_col1:
    run_stt = st.button("[1] Speech-to-Text")
    run_i2t = st.button("[2] Image-to-Text")
with action_col2:
    run_explanation = st.button("[3] Generate Explanation")
    run_english_tts = st.button("[4] Generate English Audio")
with action_col3:
    run_hindi = st.button("[5] Generate Hindi Guide")
    run_reconstruction = st.button("[6] Reconstruct Image", type="primary")

if run_stt:
    with st.spinner("Loading speech recognition and transcribing..."):
        if audio_file is not None:
            st.session_state.transcription = load_speech_to_text()(str(audio_file))["text"]
        else:
            st.session_state.transcription = sample["ground_transcript"]

if run_i2t:
    with st.spinner("Loading image captioning and analyzing the artifact..."):
        if input_image is not None:
            st.session_state.detected_visuals = load_image_to_text()(input_image)[0]["generated_text"]
        else:
            st.session_state.detected_visuals = sample["vision_caption"]

if run_explanation:
    detected_visuals = st.session_state.detected_visuals or sample["vision_caption"]
    prompt = (
        f"Historical Context: {sample['archival_context']} "
        f"Artifact Visuals: {detected_visuals}. "
        f"Era: {sample['historic_era']}. "
        f"Question: {sample['historic_question']} "
        "Provide an informative explanation:"
    )
    with st.spinner("Loading the language model and generating an explanation..."):
        st.session_state.explanation = load_nlp_explainer()(prompt, max_length=120)[0]["generated_text"]

if run_english_tts:
    if not st.session_state.explanation:
        st.warning("Generate an explanation before creating the English audio guide.")
    else:
        with st.spinner("Loading English speech synthesis and creating audio..."):
            speech_out = load_text_to_speech()(st.session_state.explanation)
            narration_path = BASE_DIR / "narration_output.wav"
            sf.write(narration_path, speech_out["audio"][0], samplerate=speech_out["sampling_rate"])
            st.session_state.narration_path = str(narration_path)

if run_hindi:
    if not st.session_state.explanation:
        st.warning("Generate an explanation before creating the Hindi guide.")
    else:
        with st.spinner("Loading translation and Hindi speech models..."):
            st.session_state.hindi_text = load_translation()(st.session_state.explanation)[0]["translation_text"]
            hindi_speech = load_hindi_tts()(st.session_state.hindi_text)
            hindi_audio_path = BASE_DIR / "hindi_audio_guide.wav"
            sf.write(hindi_audio_path, hindi_speech["audio"][0], samplerate=hindi_speech["sampling_rate"])
            st.session_state.hindi_audio_path = str(hindi_audio_path)

if run_reconstruction:
    detected_visuals = st.session_state.detected_visuals or sample["vision_caption"]
    render_prompt = (
        f"A highly detailed museum-quality reconstruction of {sample['artifact_title']}, "
        f"{sample['historic_era']}, {detected_visuals}. "
        "Accurate historical materials, realistic proportions, natural museum lighting, "
        "sharp focus, fine surface details, centered composition."
    )
    negative_prompt = (
        "blurry, low resolution, distorted, deformed, extra limbs, duplicate objects, "
        "text, watermark, logo, modern objects, oversaturated colors"
    )
    with st.spinner("Loading the image generation model and reconstructing the artifact..."):
        reconstructed_img = load_diffuser()(
            render_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=quality["steps"],
            guidance_scale=7.5,
            height=quality["size"],
            width=quality["size"],
        ).images[0]
        reconstructed_image_path = BASE_DIR / "reconstructed_artifact.png"
        reconstructed_img.save(reconstructed_image_path)
        st.session_state.reconstructed_image_path = str(reconstructed_image_path)

if st.session_state.transcription or st.session_state.detected_visuals or st.session_state.explanation:
    st.header("Multimodal Outputs")
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.subheader("Visual Analysis & Reconstruction")
        if st.session_state.detected_visuals:
            st.markdown(f"**Extracted Features (BLIP):** {st.session_state.detected_visuals}")
        if st.session_state.reconstructed_image_path:
            st.image(
                st.session_state.reconstructed_image_path,
                caption="Reconstructed Artifact (Stable Diffusion)",
                use_container_width=True,
            )

    with res_col2:
        st.subheader("Archival Reasoning & Vocalization")
        if st.session_state.transcription:
            st.markdown(f"**Transcribed Spoken Query:** {st.session_state.transcription}")
        if st.session_state.explanation:
            st.markdown(f"**Generated Explanation:** {st.session_state.explanation}")
        if st.session_state.narration_path:
            st.write("**Museum Audio Guide (English TTS):**")
            st.audio(st.session_state.narration_path)
        if st.session_state.hindi_text:
            st.markdown(f"**Hindi Translation:** {st.session_state.hindi_text}")
        if st.session_state.hindi_audio_path:
            st.write("**Multilingual Guide (Speech-to-Speech):**")
            st.audio(st.session_state.hindi_audio_path)