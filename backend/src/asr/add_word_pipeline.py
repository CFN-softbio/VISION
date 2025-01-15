from save_sentences_pickle import save_sentences_pickle
from run_tts import run_tts
from make_audio_dataset import make_audio_dataset
from src.asr.audio_utils import load_json
import os

        
if __name__ == '__main__':

    data = load_json()
    words = [entry['word'] for entry in data]
    base_dir = "/home2/common/asr_data/"
    for word in words:
    # Construct the folder path for the word
        folder_path = os.path.join(base_dir, word)
        
        # Check if the folder exists
        if os.path.exists(folder_path):
            # Skip to the next iteration if the folder exists
            continue

        print(f"Found a new word {word}")
        pickle_path = save_sentences_pickle(word, base_dir)
        run_tts(word, base_dir)

    make_audio_dataset(base_dir)



