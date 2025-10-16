import os
import time
import torch
from transformers import pipeline
from src.hal_beam_com.model_manager import ModelManager

from src.hal_beam_com.utils import set_data_fields, CogType, load_testing_model
from evaluate import load

def save_audio(data):
    audio = data[0]['voice_cog_input']
    audio_output_path = "./data/audio_input.wav"
    os.makedirs(os.path.dirname(audio_output_path), exist_ok=True)
    audio.export(audio_output_path, format="wav")

    return audio_output_path


def set_device():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    return device, torch_dtype

def load_model(base_model, finetuned, torch_dtype):
    return ModelManager.get_whisper_model(base_model, finetuned) 

def get_pipeline(model, processor, torch_dtype, device):
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=15,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device

    )

    return pipe


def invoke(data, base_model, model_number = None, word = None, finetuned=True, audio_path = None, dataset = None):

    cog = CogType.VOICE

    device, torch_dtype = set_device()

    #model_number used during testing to keep track of the number of models finetuned. Also equal to the number of finetuning datapoints - 1
    if model_number is None:
        model, processor = load_model(base_model, finetuned, torch_dtype)
        
    else:
        model, processor = load_testing_model(model_number, word, torch_dtype)

    # model, processor = load_model(base_model, finetuned, torch_dtype)
    model.to(device)

    pipe = get_pipeline(model, processor, torch_dtype, device)

    start_time = time.time()

    execution_times = []

    if dataset is not None:
        results = []

        for entry in dataset:
            audio_path = entry['audio_path']
            expected_transcription = entry['expected_transcription']
            start_time = time.time()
            result = pipe(audio_path)

            end_time = time.time()
            generated_transcription = result['text']

            wer_metric = load("wer")

            wer = wer_metric.compute(references=[expected_transcription], predictions=[generated_transcription])

            match = wer == 0

            execution_time = end_time - start_time

            execution_times.append(execution_time)

            result_data = {
                "expected_transcription": expected_transcription,
                "generated_transcription": generated_transcription.encode('utf-8'),
                "execution_time": f"{execution_time:.5f} seconds",  # Store as JSON array
                "word_error_rate": wer
            }

            results.append(result_data)

        return results, match
    
    else:

        audio_path = save_audio(data)
        result = pipe(audio_path, generate_kwargs={"task": "translate"})

        transcription = result['text']
        print(transcription)

        end_time = time.time()

        execution_time = end_time - start_time


        data[0]['bl_conf'] = 0

        data[0]['last_cog_id'] = 0

        rdb_data = {

            'model': base_model,
            'input': "",
            'output': transcription,
            'start_time': f"{start_time:.5f} seconds",
            'end_time': f"{end_time:.5f} seconds",
            'execution_time': f"{execution_time:.5f} seconds",
            'system_prompt': ""

        }

        data = set_data_fields(rdb_data, cog, data)

        return data
