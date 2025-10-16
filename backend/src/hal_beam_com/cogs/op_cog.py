#!/usr/bin/python3
import os

from src.hal_beam_com.Base import *
from src.hal_beam_com.model_manager import ModelManager
from src.hal_beam_com.utils import (
    set_data_fields,
    cog_output_fields,
    CogType,
    prompt_to_string, load_system_prompt, execute_llm_call
)


def invoke(data, base_model, finetuned=False, system_prompt_path=None, history = ""):
    cog = CogType.OP

    llm = ModelManager.get_model(base_model)

    beamline = data[0]['beamline']

    base_dir = os.path.dirname(os.path.abspath(__file__))

    print('### In process_Role {}'.format(Base().now()))

    # TODO: Shouldn't this be a boolean?
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

    operator_cog_history = data[0]['operator_cog_db_history']
    operator_cog_history += data[0]['operator_cog_history']

    prompt_template = load_system_prompt(beamline, cog, data, system_prompt_path, history = operator_cog_history)

    llm_output = execute_llm_call(llm, prompt_template, user_prompt, strip_markdown=True, history = operator_cog_history)

    data[0]['operator_cog_history'] += f"Human: {user_prompt}\n"

    end_time = time.time()
    execution_time = end_time - start_time

    prompt_as_string = prompt_to_string(prompt_template, cog=cog, save=True)

    # TODO: @Shray remove or something
    data[0]['op_cog_output'] = llm_output

    data[0]['op_cog_time'] = f"{execution_time:.5f} seconds"

    ####################

    data[0]['bl_conf'] = 0

    data[0]['termination'] = 1

    data[0]['last_cog_id'] = 3

    data[0]['operator_cog_history'] += f"AI: {llm_output}\n"

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
