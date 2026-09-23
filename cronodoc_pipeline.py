import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from PIL import Image
from datasets import Dataset, load_from_disk
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
from transformers import pipeline

base_dir = Path(__file__).resolve().parent

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f'Executing the models on device: {device}')


def safe_value(mapping, *keys, default=''):
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != '':
            return value
    return default


def build_dataset_from_csvs():
    csv_files = [
        base_dir / 'audio_speech_5k.csv',
        base_dir / 'historical_image_text_5k.csv',
        base_dir / 'instruction_qa_5k.csv',
    ]

    if not all(file.exists() for file in csv_files):
        generator = base_dir / 'generate_5k_project_datasets.py'
        if generator.exists():
            print('Missing CSV datasets detected. Generating them now...')
            subprocess.run([sys.executable, str(generator)], check=True)
        if not all(file.exists() for file in csv_files):
            raise FileNotFoundError(
                'Required datasets are missing. Please generate them in the parent NLP Projects folder.'
            )

    df_audio = pd.read_csv(csv_files[0])
    df_vision = pd.read_csv(csv_files[1])
    df_qa = pd.read_csv(csv_files[2])

    limit = 70
    merged_df = pd.DataFrame({
        'id': [f'chrono_{i:04d}' for i in range(limit)],
        'audio_filename': df_audio['audio_file'].iloc[:limit].values,
        'speaker_dialect': df_audio['dialect'].iloc[:limit].values,
        'ground_transcript': df_audio['transcript'].iloc[:limit].values,
        'image_id': df_vision['image_id'].iloc[:limit].values,
        'artifact_title': df_vision['title'].iloc[:limit].values,
        'historic_era': df_vision['era'].iloc[:limit].values,
        'vision_caption': df_vision['caption'].iloc[:limit].values,
        'user_instruction': df_qa['instruction'].iloc[:limit].values,
        'historic_question': df_qa['question'].iloc[:limit].values,
        'archival_context': df_qa['context'].iloc[:limit].values,
        'expected_answer': df_qa['answer'].iloc[:limit].values,
    })

    return Dataset.from_pandas(merged_df)


try:
    dataset_path = base_dir / 'chrono_unified_data'
    if dataset_path.exists():
        dataset = load_from_disk(str(dataset_path))
    else:
        dataset = build_dataset_from_csvs()
except Exception as exc:
    print(f'Warning: dataset load failed: {exc}')
    dataset = build_dataset_from_csvs()

sample = dataset[0]

print(f"\nProcessing ID: {safe_value(sample, 'id', 'sample_id')}")
print(f"Artifact Title: {safe_value(sample, 'artifact_title', 'title')}")
print(f"Historical Era: {safe_value(sample, 'historic_era', 'historical_era', 'era')}")

# 1) speech to text
print('\n--- [1] Speech-to-Text')
stt = pipeline('automatic-speech-recognition', model='facebook/wav2vec2-base-960h', device=device)
audio_file = safe_value(sample, 'audio_filename', 'audio_path')

if audio_file and os.path.exists(audio_file):
    transcription = stt(audio_file)['text']
else:
    transcription = safe_value(sample, 'ground_transcript', 'ground_truth_transcript', 'transcript')
print(f'Trancribed Audio: {transcription}')

# 2) image to text
print('\n--- [2] Image-to-Text')
i2t = pipeline('image-text-to-text', model='Salesforce/blip-image-captioning-base', device=device)
image_id = safe_value(sample, 'image_id')
image_file = base_dir / f'{image_id}.jpg'

if image_file.exists():
    img = Image.open(image_file)
    detected_visual = i2t(img)[0]['generated_text']
else:
    detected_visual = safe_value(sample, 'vision_caption', 'visual_caption', 'caption')
print(f'Detected Visuals: {detected_visual}')

# 3) text to explain
print("\n--- [3] Text-to-Explanation ---")
llm = pipeline(
    'text-generation',
    model='distilgpt2',
    device=device,
    tokenizer='distilgpt2'
)

prompt = (
    f"Historical Context: {safe_value(sample, 'archival_context', 'context')}. "
    f"Artifact Visuals: {detected_visual}. "
    f"Era: {safe_value(sample, 'historic_era', 'historical_era', 'era')}. "
    f"Question: {safe_value(sample, 'historic_question', 'question')}. "
    f"Provide an informative explanation."
)
result = llm(prompt, max_new_tokens=60, do_sample=True, temperature=0.8, top_p=0.9)
explanation = result[0]['generated_text'][len(prompt):].strip()
print(f"Generated Explanation: {explanation}")

# 4) Text-to-Image
print("\n--- [4] Text-to-Image ---")
try:
    t2i = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    t2i.scheduler = EulerDiscreteScheduler.from_config(t2i.scheduler.config)
    t2i = t2i.to(device)

    render_prompt = (
        f"Historical pristine reconstruction of {safe_value(sample, 'artifact_title', 'title')}, "
        f"era: {safe_value(sample, 'historic_era', 'historical_era', 'era')}, {detected_visual}, 8k museum rendering"
    )
    reconstructed_img = t2i(render_prompt, num_inference_steps=20).images[0]
    reconstructed_img.save('reconstructed_artifact.png')
    print("Visual reconstruction saved as 'reconstructed_artifact.png'!")
except Exception as exc:
    print(f'Warning: Image generation failed: {exc}')
    print('Skipping image generation step.')

# 5) text to speech
print("\n--- [5] Text-to-Speech ---")
try:
    tts = pipeline('text-to-audio', model='facebook/mms-tts-eng', device=device)
    tts_audio = tts(explanation)
    sf.write('narration_output.wav', tts_audio['audio'][0], samplerate=tts_audio['sampling_rate'])
    print("Museum narration audio saved to 'narration_output.wav'!")
except Exception as exc:
    print(f'Warning: TTS model failed: {exc}')
    print('Skipping narration generation.')

# 6) speech to speech
print("\n--- [6] Speech-to-Speech (English -> Hindi Audio) ---")
try:
    translator = pipeline('translation', model='Helsinki-NLP/opus-mt-en-hi', device=device)
    hindi_translation = translator(explanation)[0]['translation_text']
    print(f'Hindi Translation: {hindi_translation}')

    tts_hindi = pipeline('text-to-audio', model='facebook/mms-tts-hin', device=device)
    hindi_speech = tts_hindi(hindi_translation)
    sf.write('hindi_audio_guide.wav', hindi_speech['audio'][0], samplerate=hindi_speech['sampling_rate'])
    print("Hindi spoken translation saved to 'hindi_audio_guide.wav'!")
except Exception as exc:
    print(f'Warning: Hindi speech generation failed: {exc}')
    print('Skipping Hindi audio generation.')

print('\nChronodoc pipeline completed with available fallbacks.')