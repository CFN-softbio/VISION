import os
import time
from langchain_openai import AzureChatOpenAI

from src.hal_beam_com.model_manager import ModelManager
from src.hal_beam_com.utils import prepare_example_for_json, load_json_schema

selected_cog_mapping = {

    'Operator': 'Op',
    'Analysis': 'Ana',

}

from src.hal_beam_com.utils import (

    set_data_fields,
    cog_output_fields,
    load_system_prompt,
    CogType,
    convert_dict_to_string,
    load_model,
    execute_structured_llm_call

)

def invoke(data, base_model, finetuned=False, system_prompt_path=None):

    cog = CogType.REFINE

    llm = ModelManager().get_model(base_model)

    beamline = data[0]['beamline']
    selected_cog = data[0]['selected_cog']

    if data[0]['only_text_input'] == 1:
        user_prompt = data[0]['text_input']

    else:
        last_cog_id = data[0]['last_cog_id']
        cog_output_field = cog_output_fields[last_cog_id]
        user_prompt = data[0]['text_input'] + data[0][cog_output_field]

    start_time = time.time()

    json_schema = load_json_schema(beamline, cog)
    prompt_template = load_system_prompt(beamline, cog, data, system_prompt_path)

    llm_output_dict = execute_structured_llm_call(llm, prompt_template, user_prompt, json_schema)

    llm_output = convert_dict_to_string(llm_output_dict)

    end_time = time.time()

    execution_time = end_time - start_time

    llm_output_dict_updated = prepare_example_for_json(llm_output_dict, selected_cog_mapping[selected_cog])

    llm_output_string_updated = convert_dict_to_string(llm_output_dict_updated)

    llm_display_dict = {key: value for key, value in llm_output_dict_updated.items() if key != "default"}
    llm_display_dict['cog'] = selected_cog

    llm_display_output = convert_dict_to_string(llm_display_dict)

    if data[0]['only_text_input'] == 0:
        data[0]['refinement_cog_output'] = llm_output
    else:
        data[0]['add_context_tb_out'] = llm_output

    rdb_data = {

    'model': base_model,
    'input': user_prompt,
    'display_output': llm_display_output,
    'output': llm_output_string_updated,
    'output_dict': llm_output_dict_updated,
    'start_time': f"{start_time:.5f} seconds",
    'end_time': f"{end_time:.5f} seconds",
    'execution_time': f"{execution_time:.5f} seconds"

    }

    data = set_data_fields(rdb_data, cog, data)

    return data
