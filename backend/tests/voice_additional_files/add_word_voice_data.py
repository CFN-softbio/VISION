import sys
import os

import pickle
import json
from base_test_framework import BaseTestFramework
from src.hal_beam_com.utils import CogType

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.asr.save_sentences_pickle import save_sentences_pickle
from src.asr.run_tts import run_tts


word = "GISAXS"
base_dir = f"{current_dir}/asr_data/"

save_sentences_pickle(word, base_dir, testing = True)
run_tts(word, base_dir, testing = True)

wav_dir = f"{current_dir}/asr_data/{word}/audio/mp3"
transcription_file = f"{current_dir}/asr_data/{word}/sentences.pkl"

output_file = f"{current_dir}/datasets/{CogType.VOICE.value}_agent_dataset.json"

# Load transcriptions from pickle file
with open(transcription_file, "rb") as f:
    transcriptions = pickle.load(f)

# Check if output file exists; if not, create an empty file
if not os.path.exists(output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Create directories if they don't exist
    with open(output_file, "w") as f:
        json.dump([], f)  # Initialize with an empty list

# Load existing data from the file
with open(output_file, "r") as f:
    existing_data = json.load(f)

# Create new JSON objects
new_data = []
for i, transcription in enumerate(transcriptions):
    # Construct the audio file name
    audio_file_name = f"{word}_{i}.mp3"
    audio_path = os.path.join(wav_dir, audio_file_name)

    # Check if the WAV file exists
    if not os.path.exists(audio_path):
        print(f"Warning: Missing audio file for {audio_file_name}")
        continue

    # Create a dictionary for the current entry
    entry = {
        "word": word,
        "audio_path": audio_path,
        "expected_transcription": transcription,
    }
    new_data.append(entry)

# Combine new data with existing data
combined_data = existing_data + new_data

# Save the updated data back to the output file
with open(output_file, "w") as f:
    json.dump(combined_data, f, indent=4)

print(f"JSON file updated at: {output_file}")