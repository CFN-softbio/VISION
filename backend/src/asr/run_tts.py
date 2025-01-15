import os
from src.asr.audio_utils import (
    domain_specific_pronounciations, 
    text_to_speech,
    load_pickle,
    get_phonetics
)

def run_tts(word, base_dir, testing = False):
    phonetics = get_phonetics(word, testing = testing)
    sentences = load_pickle(word, base_dir)

    output_audio_folder = f"{base_dir}{word}/audio/mp3"
    os.makedirs(output_audio_folder, exist_ok=True)

    for i, sentence in enumerate(sentences):
        audio_file_name = word+"_"+str(i)+".mp3"
        output_audio_path = os.path.join(output_audio_folder, audio_file_name)
        text_to_speech(sentence, word, phonetics, output_audio_path)

    print(f"Ran TTS model and saved audio files here: {output_audio_folder}")
    return 





