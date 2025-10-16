#!/usr/bin/python3
import os

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

    print('### In {} {}'.format(cog, Base().now()))

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

    # if add_full_python_command:
    #     llm_output = "python runXs_auto.py " + llm_output
    
    # #### Test only
    # current_file_path = os.path.abspath(__file__)
    # beamline_prompts_path = os.path.join(base_dir, "../beamline_prompts/11BM/")  # Adjust as needed
    # insert_protocols(beamline_prompts_path+'runXS.py', [llm_output])
        

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


# import re

# def insert_protocols(input_filename, new_protocols, output_filename=None):
#     with open(input_filename, 'r') as f:
#         content = f.read()

#     # Pattern to locate the protocols block
#     pattern = r'protocols\s*=\s*\[(.*?)\](?=\s*#|\s*\n|$)'  # Non-greedy match of content inside brackets

#     # Function to append new protocols
#     def replacer(match):
#         existing = match.group(1).strip()
#         if existing:
#             # Add comma if necessary
#             updated = existing + '\n' + ',\n    '.join(new_protocols) + ',\n    '
#         else:
#             updated = '\n   ' + '\n    '.join(new_protocols) + ',\n    '
#         return f'protocols = [\n    {updated}\n    ]'

#     # Replace protocols block
#     new_content = re.sub(pattern, replacer, content, flags=re.DOTALL)

#     # Determine the output filename
#     if output_filename is None:
#         base_name, ext = os.path.splitext(input_filename)  # Split into base name and extension
#         output_filename = f"{base_name}_llm{ext}"  # Append '_llm' before the extension

#     # Write updated content to the new file
#     with open(output_filename, 'w') as f:
#         f.write(new_content)

#     print(f"Inserted {len(new_protocols)} protocol(s) into {output_filename}.")

#     # # Example usage
#     # insert_protocols('runXS.py', [
#     #     "Protocols.circular_average(ylog=True, plot_range=[0, 0.12, None, None], label_filename=True)",
#     #     "Protocols.linecut_angle(q0=0.01687, dq=0.00455*1.5, show_region=False)"
#     # ])
