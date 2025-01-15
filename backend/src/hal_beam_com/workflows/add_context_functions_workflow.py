import os

from src.hal_beam_com.cogs import voice_cog, refinement_cog
from src.hal_beam_com.utils import tab3_response

from src.hal_beam_com.utils import append_to_command_examples

import os


def run(data, queue, audio_base_model, audio_finetuned, text_base_model, text_finetuned):

    if data[0]['only_text_input'] == 0:

        data = voice_cog.invoke(data, base_model = audio_base_model, finetuned = audio_finetuned)
    
    data = refinement_cog.invoke(data, text_base_model, text_finetuned)

    return data

