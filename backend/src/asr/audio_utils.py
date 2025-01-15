import os
import pickle
import json
from gtts import gTTS
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def domain_specific_pronounciations(text, word, phonetics):
    text = text.replace(word, phonetics)
    return text

def text_to_speech(text, word, phonetics, output_audio_path):
    """Convert text to speech and save as an audio file."""
    text = domain_specific_pronounciations(text, word, phonetics)
    tts = gTTS(text=text, lang='en')
    tts.save(output_audio_path)

def load_pickle(word, base_dir):
    
    path = f"{base_dir}{word}/sentences.pkl"
    with open(path, "rb") as file:
        data = pickle.load(file)

    return data

def load_json():
    json_file = "./words.json"
    with open(json_file, "r") as file:
        data = json.load(file)

    return data

def get_phonetics(search_word, testing = False):

   if testing != True:
       json_file = "./words.json"
   else:
       json_file = "../src/asr/words.json"
   try:
    # Load the JSON file
        with open(json_file, "r") as file:
            data = json.load(file)

    # Ensure the JSON file contains a list
        if isinstance(data, list):
            # Search for the word in the list of dictionaries
            found = False
            for entry in data:
                if entry.get("word") == search_word:
                    print(f"Phonetics for '{search_word}': {entry.get('phonetics')}")
                    found = True
                    return entry.get('phonetics')

            if not found:
                print(f"The word '{search_word}' does not exist in the JSON file.")
        else:
            print("Error: JSON data is not a list of dictionaries.")

   except FileNotFoundError:
        print(f"Error: The file '{json_file}' was not found.")
   except json.JSONDecodeError:
        print(f"Error: Failed to decode the JSON file '{json_file}'. Is it valid JSON?")
   except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
def prepare_dataset(batch, feature_extractor, tokenizer):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]


    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

def compute_metrics(pred, tokenizer, metric):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

def populate_sentences(word, sentences):
    return [sentence.format(word=word) for sentence in sentences]

def is_audio_in_length_range(length):
    return length > min_input_length and length < max_input_length

def is_labels_in_length_range(labels):
    return len(labels) < max_label_length