"""
Evaluation Workflow - Evaluates simulation results against user intent

This workflow uses an LLM to evaluate whether:
1. The generated code correctly implements the user's query
2. The PV changes match what the code was supposed to produce
3. The overall pipeline (query → code → PV changes) was successful
"""

from typing import List, Dict
from jinja2 import Environment, FileSystemLoader
import os

from src.hal_beam_com.utils import (
    load_model,
    execute_llm_call,
    get_model_config,
    create_dynamic_op_prompt,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage


def run(data: List[Dict], **kwargs) -> List[Dict]:
    """
    Evaluate simulation results using an LLM.
    
    Expected in data[0]:
        - original_query: The user's original command/query
        - generated_code: The code that was generated and simulated
        - pv_events: List of PV event dictionaries from the simulation
        - sim_id: Simulation ID
        
    Returns:
        data with 'evaluation' field containing the LLM's assessment
    """
    
    # Extract required fields
    original_query = data[0].get('original_query', '')
    generated_code = data[0].get('generated_code', '')
    pv_events = data[0].get('pv_events', [])
    sim_id = data[0].get('sim_id', '')
    beamline = data[0].get('beamline', '11BM')
    user_id = data[0].get('user_id', 'guest')
    
    # Filter PV events to only include 'pv' type events
    pv_events = [evt for evt in pv_events if evt.get('type') == 'pv']
    
    # Render the op_cog prompt to get available commands for context
    op_cog_commands = create_dynamic_op_prompt(beamline, user_id)
    
    # Load the Jinja2 template
    template_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "cogs", "prompt_templates", "jinja_templates"
    )
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("evaluation_prompt.j2")
    
    # Render the prompt with op_cog commands included
    evaluation_prompt = template.render(
        original_query=original_query,
        generated_code=generated_code,
        pv_events=pv_events,
        op_cog_commands=op_cog_commands
    )
    
    # Load the LLM model
    model_name = get_model_config("text", "pv_evaluator")
    llm = load_model(model_name)
    
    # Create a prompt template matching the pattern in utils.py
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=evaluation_prompt),
        ("human", "{text}")
    ])
    
    # Execute LLM call
    evaluation_result = execute_llm_call(
        llm=llm,
        prompt_template=prompt_template,
        user_prompt="Please provide your evaluation.",
        strip_markdown=False
    )
    
    # Store results in data with proper response format for frontend
    data[0]['type'] = 'evaluation_result'
    data[0]['evaluation'] = evaluation_result
    data[0]['status'] = 'success'
    
    return data
