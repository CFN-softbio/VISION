#!/usr/bin/python3
import os

from langchain_openai import AzureChatOpenAI
from src.hal_beam_com.Base import *
from src.hal_beam_com.model_manager import ModelManager
from src.hal_beam_com.utils import (

    cog_output_fields,
    set_data_fields,
    SystemPromptType,
    CogType,
    process_llm_output,
    load_model,
    prompt_to_string, load_system_prompt, execute_llm_call

)

def invoke(data, base_model, finetuned=False, system_prompt_type=SystemPromptType.LIST_OUTPUT, system_prompt_path=None, testing = False):
    cog = CogType.CLASSIFIER

    llm = ModelManager.get_model(base_model)

    start_time = time.time()
    beamline = data[0]['beamline']  # Index by numbers is not informative, but re-write would require major refactoring

    base_dir = os.path.dirname(os.path.abspath(__file__))

    prompt_template = load_system_prompt(beamline, cog, data, system_prompt_path, system_prompt_type = system_prompt_type, testing = testing)

    if data[0]['only_text_input'] == 1:
        user_prompt = data[0]['text_input']
        print(user_prompt)

    else:
        last_cog_id = data[0]['last_cog_id']
        cog_output_field = cog_output_fields[last_cog_id]

        user_prompt = data[0]['text_input'] + data[0][cog_output_field]

    llm_output = execute_llm_call(llm, prompt_template, user_prompt)

    next_cog = process_llm_output(llm_output, system_prompt_type)

    end_time = time.time()
    execution_time = end_time - start_time

    prompt_as_string = prompt_to_string(prompt_template, cog=cog, save=True)

    data[0]['next_cog'] = next_cog

    data[0]['classifier_cog_output'] = next_cog

    print(data[0]['classifier_cog_output'])

    data[0]['last_cog_id'] = 4

    rdb_data = {

        'model': base_model,
        'input': user_prompt,
        'output': next_cog,
        'start_time': f"{start_time:.5f} seconds",
        'end_time': f"{end_time:.5f} seconds",
        'execution_time': f"{execution_time:.5f} seconds",
        'system_prompt': prompt_as_string

    }

    # TODO: Why does this get converted to a list? Then we have to index by 0 again, while it contains a dict.
    data = set_data_fields(rdb_data, cog, data)

    return data
