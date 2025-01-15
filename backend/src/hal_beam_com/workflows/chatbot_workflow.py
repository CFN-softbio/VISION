from src.hal_beam_com.cogs import voice_cog
from src.hal_beam_com.chatbot_utils import (

    scientist,
    classifier,
    generalist,
    scientist,
    beamline_scientist

)

def run(data, queue, audio_base_model, audio_finetuned, text_base_model, text_finetuned):
    if data[0]['only_text_input'] == 0:
        data = voice_cog.invoke(data, base_model=audio_base_model, finetuned=audio_finetuned)

    prompt = data[0]['text_input'] + data[0]['voice_cog_output']

    data[0]['prompt'] = prompt

    # prompt_type = use_semantic_router(prompt)

    prompt_type = classifier(prompt, history=data[0]['history'])

    match prompt_type:
        case "Generalist":
            print("GOING TO GENERALIST")
            bot_response = generalist(prompt, history=data[0]['history'])

        case "Scientist":
            print("GOING TO SCIENTIST")
            bot_response = scientist(prompt, history=data[0]['history'], paper_directory="/home2/smathur/RAG/CFN_publication_PDFs")

        case "Beamline":
            print("GOING TO BEAMLINE SCIENTIST")
            bot_response = beamline_scientist(prompt, history=data[0]['history'], paper_directory="/home2/smathur/RAG/Beamline_manual")


    data[0]['chatbot_response'] = bot_response
    data[0]['history'] += f"\nHuman: {prompt} \n AI: {data[0]['chatbot_response']}"

    print(data)
    return data
