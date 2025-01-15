'''
Make an audio-text dataset with adding one example at a time (so 48 datasets)
finetune whisper each time (so 48 finetunes)
evalute on test set each time 
'''

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from utils_audio_testing import cumulative_finetuning

import pickle

from src.asr.save_sentences_pickle import save_sentences_pickle
from src.asr.run_tts import run_tts

base_dir = "/home2/common/asr_data"
# word = "bioSAXS"
# audio_folder = f"{base_dir}/{word}/audio/mp3/"
# transcription_file = f"{base_dir}/{word}/sentences.pkl"

# output_wav_folder = f"{base_dir}/{word}/audio/wav_testing"

# with open(transcription_file, "rb") as f:
#     transcriptions = pickle.load(f)

# cumulative_finetuning(word, output_wav_folder, audio_folder, transcriptions)

words = ['gpCAM', 'SAXS', 'WAXS', 'GISAXS', 'GIWAXS', 'SciAnalysis', 'bioSAXS']

j = 0
for word in words:
    audio_folder = f"{base_dir}/{word}/audio/mp3/"
    transcription_file = f"{base_dir}/{word}/sentences.pkl"

    output_wav_folder = f"{base_dir}/{word}/audio/wav_testing"

    with open(transcription_file, "rb") as f:
        transcriptions = pickle.load(f)

    cumulative_finetuning(word, output_wav_folder, audio_folder, transcriptions, j)
    j += 48

