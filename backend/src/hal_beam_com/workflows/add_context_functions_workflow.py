from src.hal_beam_com.cogs import voice_cog, refinement_cog


# def run(data, queue, audio_base_model, audio_finetuned, text_base_model, text_finetuned):
def run(data, audio_base_model, audio_finetuned, text_base_model, text_finetuned):

    if data[0]['only_text_input'] == 0:

        data = voice_cog.invoke(data, base_model = audio_base_model, finetuned = audio_finetuned)
    
    data = refinement_cog.invoke(data, text_base_model, text_finetuned)

    return data

