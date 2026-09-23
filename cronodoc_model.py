from pathlib import Path
import pandas as pd
from datasets import Dataset

base_dir = Path(__file__).resolve().parent.parent

df_audio = pd.read_csv('audio_speech_5k.csv')
df_vision = pd.read_csv('historical_image_text_5k.csv')
df_qa = pd.read_csv('instruction_qa_5k.csv')

limit = 70

merged_df = pd.DataFrame({
    'id': [f'chrono_{i:04d}' for i in range(limit)],

    # audio field
    'audio_filename': df_audio['audio_file'].iloc[:limit].values,
    'speaker_dialect': df_audio['dialect'].iloc[:limit].values,
    'ground_transcript': df_audio['transcript'].iloc[:limit].values,

    # image field
    'image_id': df_vision['image_id'].iloc[:limit].values,
    'artifact_title': df_vision['title'].iloc[:limit].values,
    'historic_era': df_vision['era'].iloc[:limit].values,
    'vision_caption': df_vision['caption'].iloc[:limit].values,

    # QA field
    'user_instruction': df_qa['instruction'].iloc[:limit].values,
    'historic_question': df_qa['question'].iloc[:limit].values,
    'archival_context': df_qa['context'].iloc[:limit].values,
    'expected_answer': df_qa['answer'].iloc[:limit].values
})

dataset = Dataset.from_pandas(merged_df)
dataset.save_to_disk('chrono_unified_data')
print('Successfully merged and saved datasets to CHRONO Unified data folder !!!!')

