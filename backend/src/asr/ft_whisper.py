import os
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, pipeline
from datasets import load_dataset, DatasetDict, load_from_disk, Audio
import torch
from peft import PeftModel
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

from src.asr.audio_utils import (

    prepare_dataset,
    DataCollatorSpeechSeq2SeqWithPadding
)

import datetime
import time

def is_audio_in_length_range(length):
    return length > min_input_length and length < max_input_length

def is_labels_in_length_range(labels):
    return len(labels) < max_label_length

hf_username = "shray98"
# word = 'gpcam'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
base_model = 'whisper-large-v3'
model_name_or_path = "openai/"+base_model

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

peft_model_id = f"shray98/peft-{timestamp}-openai-{base_model}"
task = "transcribe"

base_dir = "/home2/common/asr_data/"
# base_dir = "./data/"
dataset_timestamp = "20241126_145426"
dataset_name = f"{base_dir}{dataset_timestamp}_audio_dataset"

model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, device_map='auto') 
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, task=task)
processor = WhisperProcessor.from_pretrained(model_name_or_path, task=task)

dataset = load_from_disk(dataset_name)

cfn_prop_audio = DatasetDict()
cfn_prop_audio['train'] = dataset['train']
cfn_prop_audio['test']=dataset['test']

cfn_prop_audio = cfn_prop_audio.map(prepare_dataset, remove_columns=cfn_prop_audio.column_names["train"], num_proc=1)

max_input_length = 30
min_input_length = 0

cfn_prop_audio = cfn_prop_audio.filter(
    is_audio_in_length_range, num_proc=1, input_columns=["input_length"]
)

max_label_length = model.config.max_length

cfn_prop_audio = cfn_prop_audio.filter(
    is_labels_in_length_range, num_proc=1, input_columns=["labels"]
)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

metric = evaluate.load("wer")

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

model = get_peft_model(model, config)
model.print_trainable_parameters()

training_args = Seq2SeqTrainingArguments(
    output_dir="temp",  # change to a repo name of your choice
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-3,
    warmup_steps=50,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    fp16=True,
    per_device_eval_batch_size=8,
    generation_max_length=128,
    logging_steps=25,
    remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
    label_names=["labels"], 
     # same reason as above
)


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=cfn_prop_audio["train"],
    eval_dataset=cfn_prop_audio["test"],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

start_time = time.time()
trainer.train()
execution_time = time.time() - start_time
print(f"Training completed in {execution_time:.2f} seconds.")

model.push_to_hub(peft_model_id)

print(f"Saved finetuned model here: {peft_model_id}")