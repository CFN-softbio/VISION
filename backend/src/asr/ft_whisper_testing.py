'''This file is used with the testing framework 

to finetune whisper'''

import os
import time
from transformers import (
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from peft import LoraConfig, get_peft_model

from src.asr.audio_utils import (

    prepare_dataset,
    DataCollatorSpeechSeq2SeqWithPadding
)

# Define helper functions and classes
def is_audio_in_length_range(length, min_input_length=0, max_input_length=30):
    return min_input_length < length < max_input_length

def is_labels_in_length_range(labels, max_label_length):
    return len(labels) < max_label_length

# Main function
def finetune_model(cfn_prop_audio, i, save_location = 'local'):
    hf_username = "shray98"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    base_model = "whisper-large-v3"
    model_name_or_path = f"openai/{base_model}"
    peft_model_id = f"{hf_username}/peft-cumulative-testing-all-{i}-openai-{base_model}"
    task = "transcribe"

    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, device_map="auto")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
    tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, task=task)
    processor = WhisperProcessor.from_pretrained(model_name_or_path, task=task)

    # Preprocessing dataset
    cfn_prop_audio = cfn_prop_audio.map(
        lambda batch: prepare_dataset(batch, feature_extractor, tokenizer),
        remove_columns=cfn_prop_audio.column_names["train"],
        num_proc=1,
    )

    max_input_length = 30
    min_input_length = 0
    max_label_length = model.config.max_length

    cfn_prop_audio = cfn_prop_audio.filter(
        lambda example: is_audio_in_length_range(example["input_length"], min_input_length, max_input_length),
        num_proc=1,
    )
    cfn_prop_audio = cfn_prop_audio.filter(
        lambda example: is_labels_in_length_range(example["labels"], max_label_length),
        num_proc=1,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    training_args = Seq2SeqTrainingArguments(
        output_dir="temp",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        warmup_steps=50,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        fp16=True,
        per_device_eval_batch_size=8,
        generation_max_length=128,
        logging_steps=25,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=cfn_prop_audio["train"],
        eval_dataset=cfn_prop_audio["test"],
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )
    model.config.use_cache = False

    start_time = time.time()
    trainer.train()
    execution_time = time.time() - start_time
    
    print(f"Training completed in {execution_time:.2f} seconds.")

    if save_location == 'local':
        save_path = f'{hf_username}/{peft_model_id}'

# Check if the path exists, if not, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Save the model
        model.save_pretrained(save_path)
        print(f"Saved finetuned model here: {save_path}")

    elif save_location == 'hugging_face':
        model.push_to_hub(peft_model_id)
        print(f"Saved finetuned model here: {peft_model_id}")
    

    return execution_time
