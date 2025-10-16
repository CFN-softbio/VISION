import time
import json
import re
from src.hal_beam_com.model_manager import ModelManager

from src.hal_beam_com.utils import (


    set_data_fields,
    parse_verifier_json,
    SystemPromptType,
    CogType,
    prompt_to_string, load_system_prompt, execute_llm_call

)

def invoke(data, base_model, finetuned=False, system_prompt_type=SystemPromptType.LIST_OUTPUT, system_prompt_path=None, testing = False, history = ""):
    cog = CogType.VERIFIER
    llm = ModelManager.get_model(base_model)
    start_time = time.time()

    beamline = data[0]['beamline']
    prompt_template = load_system_prompt(beamline, cog, data, system_prompt_path)
    response = execute_llm_call(llm, prompt_template, user_prompt = "Answer:", history = data[0]['classifier_cog_history'])
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"VERIFIER EXECUTION TIME: {execution_time:.5f} seconds")

    verification_result = parse_verifier_json(response)
    
    # Add verification results to data
    data[0]['verification_result'] = verification_result
    data[0]['verification_flags'] = []
    
    # Set flags for any issues found
    if verification_result['missing_info']:
        data[0]['verification_flags'].append('MISSING_INFO')
    if verification_result['hallucinations']:
        data[0]['verification_flags'].append('HALLUCINATION')
    if not verification_result['is_response_justified']:
        data[0]['verification_flags'].append('UNJUSTIFIED_RESPONSE')

    data[0]['verifier_cog_output'] = response

    data[0]['last_cog_id'] = 5

    data[0]['verifier_cog_time'] = f"{execution_time:.5f} seconds"

    prompt_as_string = prompt_to_string(prompt_template, cog=cog, save=True)

    rdb_data = {

        'model': base_model,
        'input': "",
        'output': response,
        'start_time': f"{start_time:.5f} seconds",
        'end_time': f"{end_time:.5f} seconds",
        'execution_time': f"{execution_time:.5f} seconds",
        'system_prompt': prompt_as_string

    }

    data = set_data_fields(rdb_data, cog, data)
    
    return data

