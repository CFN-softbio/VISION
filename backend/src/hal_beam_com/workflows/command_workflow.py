import os
from src.hal_beam_com.cogs import voice_cog, op_cog, analysis_cog, classifier_cog_semantic_router, \
    classifier_cog, verification_cog

from src.hal_beam_com.utils import SystemPromptType, get_model_config, id_to_cog


# def run(data, queue, audio_base_model=None, audio_finetuned=False, text_base_model=None, text_finetuned=False, use_static_prompt=False):
def run(data, audio_base_model=None, audio_finetuned=False, text_base_model=None, text_finetuned=False, use_static_prompt=False):
    # Use configured models if none specified
    if audio_base_model is None:
        audio_base_model = get_model_config("audio", "transcription")
    if text_base_model is None:
        text_base_model = get_model_config("text", "default")
    # Invoke the voice cog
    if data[0]['only_text_input'] == 0:
        data = voice_cog.invoke(data, base_model=audio_base_model, finetuned=audio_finetuned)

        # Invoke the classifier cog

    # data = classifier_cog_semantic_router.invoke(data, base_model=text_base_model, finetuned=False)

    data = classifier_cog.invoke(data, base_model=get_model_config("text", "classifier"), finetuned=text_finetuned,
                                   system_prompt_type=SystemPromptType.ONE_WORD_OUTPUT, system_prompt_path=None)

    data[0]['only_text_input'] = 0

    # TODO: Make this match case and use enums

    next_cog = data[0]['next_cog']

    # Call either the Op cog, the Ana cog or the notebook work flow depending on what the classifier cog said
    match next_cog:
        case "Op":
            if use_static_prompt:
                current_file_path = os.path.abspath(__file__)
                system_prompt_path = os.path.join(os.path.dirname(current_file_path),
                                                  '../cogs/prompt_templates/11BM/operator_cog/op_cog.txt')
                data = op_cog.invoke(data,
                                       base_model=get_model_config("text", "operator"),
                                       finetuned=text_finetuned,
                                       system_prompt_path=system_prompt_path)
            else:
                data = op_cog.invoke(data,
                                       base_model=get_model_config("text", "operator"),
                                       finetuned=text_finetuned)
                data[0]['final_output'] = data[0]['operator_cog_output']
                

        case "Ana":
            data = analysis_cog.invoke(data, base_model=text_base_model, finetuned=text_finetuned)
            print(data[0]['ana_cog_output'])

        case "Notebook":
            data[0]['final_output'] = next_cog
            return data
    
    if data[0]['include_verifier']:
        data = verification_cog.invoke(
                data,
                base_model = get_model_config("text", "verifier"),
                finetuned = text_finetuned,
            )

        if data[0]['verification_flags']:
            # Handle issues found during verification
            print("Verification flags:", data[0]['verification_flags'])
        print(data[0]['op_cog_output'])

    return data
