#!/usr/bin/python3
import os
import re

from langchain_openai import AzureChatOpenAI
from src.hal_beam_com.Base import *
from src.hal_beam_com.model_manager import ModelManager
from src.hal_beam_com.utils import (

    set_data_fields,
    cog_output_fields,
    CogType,
    load_model,
    prompt_to_string, load_system_prompt, execute_llm_call

)


def invoke(data, base_model, finetuned=False, system_prompt_path=None, add_full_python_command=True):

    cog = CogType.ANA

    #llm_output, prompt_as_string = run_llm(data, cog, base_model, system_prompt_path)

    llm = ModelManager.get_model(base_model)

    beamline = data[0]['beamline']

    base_dir = os.path.dirname(os.path.abspath(__file__))

    prompt_template = load_system_prompt(beamline, cog, data, system_prompt_path)

    print('### In process_Role {}'.format(Base().now()))

    if data[0]['only_text_input'] == 1:
        user_prompt = data[0]['text_input']
        print(user_prompt)

    else:
        last_cog_id = 0
        # Assuming the last cog is the voice cog for now
        cog_output_field = cog_output_fields[last_cog_id]

        user_prompt = data[0]['text_input'] + data[0][cog_output_field]

    start_time = time.time()

    print(user_prompt)
    llm_output = execute_llm_call(llm, prompt_template, user_prompt, strip_markdown=True)

    if add_full_python_command:
        llm_output = "python runXs_auto.py " + llm_output

    end_time = time.time()
    execution_time = end_time - start_time

    prompt_as_string = prompt_to_string(prompt_template, cog=cog, save=True)

    data[0]['ana_cog_output'] = llm_output

    data[0]['ana_cog_time'] = f"{execution_time:.5f} seconds"

    ####################

    data[0]['bl_conf'] = 0

    data[0]['termination'] = 1

    data[0]['last_cog_id'] = 5

    rdb_data = {
        'model': base_model,
        'input': user_prompt,
        'output': llm_output,
        'start_time': f"{start_time:.5f} seconds",
        'end_time': f"{end_time:.5f} seconds",
        'execution_time': f"{execution_time:.5f} seconds",
        'system_prompt': prompt_as_string
    }

    data = set_data_fields(rdb_data, cog, data, debug_printing=False)

    return data
