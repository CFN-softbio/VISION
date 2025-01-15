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

import sys
import time 

import datetime
from src.asr.ft_whisper_testing import finetune_model

def make_audio_dataset(transcriptions_train, transcriptions_test, transcription_data):

    cfn_prop_tokenized_audio = DatasetDict()

    transcriptions_train.extend(transcription_data)
    transcriptions_test.extend(transcription_data)

    # print(f"Added {transcription_data} to the finetuning dataset. Wav files stored here {output_wav_folder}")

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

    return cfn_prop_tokenized_audio

def get_audio_text_pair(transcription_data, sentences, audio_folder, word, output_wav_folder, i):

    audio_name = word+"_"+str(i)+".mp3"
    audio_file_path = os.path.join(audio_folder, audio_name)

    text = sentences[i]

    audio_segment = AudioSegment.from_mp3(audio_file_path)

    output_file_name = audio_name[:-4]+'.wav'

    output_file_path = os.path.join(output_wav_folder, output_file_name)

    if not os.path.exists(output_wav_folder):
        os.makedirs(output_wav_folder)

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

transcriptions_train = []
transcriptions_test = []

def cumulative_finetuning(word, output_wav_folder, audio_folder, sentences, j = 0):
   
   finetuning_times = []
   for i in range(len(os.listdir(audio_folder))):
        
        transcription_data = []
        transcription_data = get_audio_text_pair(transcription_data, sentences, audio_folder, word, output_wav_folder, i)


        print(len(transcription_data))

        cfn_prop_tokenized_audio = make_audio_dataset(transcriptions_train, transcriptions_test, transcription_data) 

        print(cfn_prop_tokenized_audio)

        finetuning_time = finetune_model(cfn_prop_tokenized_audio, str(i+j), word)

        pickle_file_path = f'./finetuning_times_entire.pkl'

        # Check if the file exists and load its content
        if os.path.exists(pickle_file_path):
            with open(pickle_file_path, "rb") as file:
                finetuning_times = pickle.load(file)  # Load existing data
        else:
            finetuning_times = []  # Initialize as an empty list if file doesn't exist

        # Append the current value to the list
        finetuning_times.append(finetuning_time)
        
        # Save the updated list back to the file
        with open(pickle_file_path, "wb") as file:
            pickle.dump(finetuning_times, file)

        print(f"Added finetuning time to pickle file: {pickle_file_path} for run {i+j}")


