import os
import pickle
import os
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile
from scipy.signal import resample
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
import re
from src.asr.audio_utils import (

    load_pickle,
    load_json

)
import datetime

def load_audio_text(word, output_wav_folder, audio_folder, sentences):

   transcription_data = []
   for i in range(len(os.listdir(audio_folder))):

        audio_name = word+"_"+str(i)+".mp3"
        audio_file_path = os.path.join(audio_folder, audio_name)

        text = sentences[i]

        audio_segment = AudioSegment.from_mp3(audio_file_path)

        output_file_name = audio_name[:-4]+'.wav'

        output_file_path = os.path.join(output_wav_folder, output_file_name)

        audio_segment.export(output_file_path, format='wav')

        sampling_rate, audio_array = wavfile.read(output_file_path)
        
        # Resample the audio to 16000 Hz if necessary
        target_sampling_rate = 16000
        if sampling_rate != target_sampling_rate:
            number_of_samples = int(len(audio_array) * target_sampling_rate / sampling_rate)
            audio_array = resample(audio_array, number_of_samples)
            sampling_rate = target_sampling_rate
        
        # Ensure the audio array is of an appropriate data type
        if audio_array.dtype == np.int64:
            audio_array = audio_array.astype(np.int32)
        
        # Create the audio dictionary
        audio_dict = {
            'path': output_file_path,
            'array': audio_array.tolist(),  # Convert numpy array to list for serialization
            'sampling_rate': sampling_rate,
        }
        
        # Add the transcription data to the list
        transcription_data.append({
            'audio': audio_dict,
            'sentence': text,
        })
    
   return transcription_data

def make_audio_dataset(base_dir):

    data = load_json()
    words = [entry['word'] for entry in data]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'{base_dir}{timestamp}_audio_dataset'

    transcriptions_train = []
    transcriptions_test = []

    for word in words:

        sentences = load_pickle(word, base_dir)

        print(sentences[0])

        mp3_folder = f"{base_dir}{word}/audio/mp3"
        output_wav_folder = f"{base_dir}{word}/audio/wav"
        # output_wav_folder = f"../data/{word}/audio/wav"
        os.makedirs(output_wav_folder, exist_ok=True)

        transcription_data = load_audio_text(word, output_wav_folder, mp3_folder, sentences)

        cfn_prop_tokenized_audio = DatasetDict()

        transcriptions_train.extend(transcription_data)
        transcriptions_test.extend(transcription_data)

        print(f"Added {word} to the finetuning dataset. Wav files stored here {output_wav_folder}")

    cfn_prop_audio_train = Dataset.from_dict({
        'audio': [item['audio'] for item in transcriptions_train],
        'sentence': [item['sentence'] for item in transcriptions_train],
    })

    cfn_prop_audio_test = Dataset.from_dict({
        'audio': [item['audio'] for item in transcriptions_test],
        'sentence': [item['sentence'] for item in transcriptions_test],
    })

    cfn_prop_tokenized_audio['train'] = cfn_prop_audio_train
    cfn_prop_tokenized_audio['test'] = cfn_prop_audio_test

    cfn_prop_tokenized_audio.save_to_disk(output_file)

    print(f"Saved audio dataset here: {output_file}")

    return 

