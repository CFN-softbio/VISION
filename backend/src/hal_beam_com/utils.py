import os
import re
from enum import Enum
import json
import ast
import numpy as np
import importlib
from anthropic import Anthropic
import abacusai

from src.hal_beam_com.file_utils import get_command_dir

AbacusClient = abacusai.ApiClient
import torch
from jinja2 import Environment, FileSystemLoader
from collections import defaultdict
from functools import lru_cache

# generic exponential-backoff helper
from tenacity import retry, wait_exponential, stop_after_attempt

# AWS Bedrock support (optional dependency)
try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:          # keep the module import-safe on systems w/o boto3
    boto3, ClientError = None, None

# Google Gemini support (optional dependency)
try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:          # keep the module import-safe on systems w/o google-genai
    genai = None
    genai_types = None

from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.llms import Ollama
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from peft import PeftModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


class SystemPromptType(Enum):
    LIST_OUTPUT = "list_output"
    ONE_WORD_OUTPUT = "one_word_output"
    ID_OUTPUT = "id_output"


class InputChannel(Enum):
    COMMAND = "command"
    ADD_CONTEXT = "add_context"
    CONFIRM_CONTEXT = "confirm_context"
    CHATBOT = "chatbot"
    CONFIRM_CODE = "confirm_code"
    SIMULATE       = "simulate"          # user pressed the *Simulate* button
    SIMULATE_ABORT = "simulate_abort"    # front-end abort message
    EVALUATE_SIMULATION = "evaluate_simulation"  # evaluate simulation results
    REGISTER = "register"
    LOGIN = "login"
    LOGOUT = "logout"

class CogType(Enum):
    VOICE = "voice"
    CLASSIFIER = "classifier"
    OP = "operator"
    ANA = "analysis"
    REFINE = "refinement"
    GPCAM = "gpcam"
    XICAM = "xicam"
    VERIFIER = "verifier"


class LLMProviders(Enum):
    OLLAMA = "ollama"
    OLLAMA_CHAT = "ollama_chat"
    HUGGING_FACE = "hugging_face"
    AZURE = "azure"
    CLAUDE = "claude"
    ABACUS = "abacus"
    AWS_BEDROCK = "aws_bedrock"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"

class BedrockLLM:
    """Tiny wrapper that just carries the boto3 client + model-ARN."""
    def __init__(self, client, model_id):
        self.client   = client
        self.model_id = model_id

class GeminiLLM:
    """Tiny wrapper that carries the Google Gemini client + model name."""
    def __init__(self, client, model_name):
        self.client     = client
        self.model_name = model_name


base_prompts = {

    CogType.CLASSIFIER: 'classifier_cog',
    CogType.OP: 'op_cog',
    CogType.REFINE: 'refinement_cog',
    CogType.ANA: 'analysis_cog',
    CogType.VERIFIER: 'verifier_cog'

}

base_prompts_testing = {

    (CogType.CLASSIFIER, SystemPromptType.ID_OUTPUT): 'classifier_cog_id',
    (CogType.CLASSIFIER, SystemPromptType.ONE_WORD_OUTPUT): 'classifier_cog_one_word',
    (CogType.CLASSIFIER, SystemPromptType.LIST_OUTPUT): 'classifier_cog_list',

}

cog_output_fields = {
    0: f"{CogType.VOICE.value}_cog_output",
    1: 'text_output',
    2: 'next_cog',
    3: f"{CogType.OP.value}_cog_output",
    4: f"{CogType.CLASSIFIER.value}_cog_output",
    5: f"{CogType.ANA.value}_cog_output",
    6: f"{CogType.REFINE.value}_cog_output",
    7: f"{CogType.VERIFIER.value}_cog_output"
}

id_to_cog = {
    0: "Op",
    1: "Ana",
    2: "Notebook",
    3: "gpcam",
    4: "xicam",
    5: "verifier"
}

base_models_path = {
    'mistral': {
        'model': 'mistral',
        'provider': LLMProviders.OLLAMA,
    },
    # 'Llama-3_1-Nemotron-51B-instruct': {
    #     'model': "nvidia/Llama-3_1-Nemotron-51B-Instruct",
    #     'provider': LLMProviders.HUGGING_FACE,
    # },
    'qwen2.5-coder': {
        'model': "qwen2.5-coder:32b",
        'provider': LLMProviders.OLLAMA,
    },
    'claude-3.5-sonnet': {  # 20241022, retired now
        'model': "claude-3-5-sonnet-20241022",
        'provider': LLMProviders.CLAUDE,
    },
    'claude-4.5-sonnet': {
        'model': "claude-sonnet-4-5-20250929",
        'provider': LLMProviders.CLAUDE,
    },
    'qwen2': {
        'model': "qwen2",
        'provider': LLMProviders.OLLAMA,
    },
    'qwen2.5': {
        'model': "qwen2.5",
        'provider': LLMProviders.OLLAMA,
    },
    'mistral-nemo': {
        'model': "mistral-nemo",
        'provider': LLMProviders.OLLAMA,
    },
    'gpt-4o': {
        'model': "gpt-4o",
        'provider': LLMProviders.AZURE,
    },
    'phi3.5': {
        'model': "phi3.5",
        'provider': LLMProviders.OLLAMA
    },
    'phi3.5-fp16': {
        'model': "phi3.5:3.8b-mini-instruct-fp16",
        'provider': LLMProviders.OLLAMA
    },
    'athene-v2': {
        'model': "athene-v2",
        'provider': LLMProviders.OLLAMA
    },
    'athene-v2-agent': {
        'model': "hf.co/lmstudio-community/Athene-V2-Agent-GGUF",
        'provider': LLMProviders.OLLAMA
    },
    'llama3.3': {
        'model': "llama3.3",
        'provider': LLMProviders.OLLAMA
    },
    # # TODO: Here stop the old models
    # # 'deepseek-r1-1.5b': {
    # #     'model': "deepseek-r1:1.5b",
    # #     'provider': LLMProviders.OLLAMA
    # # },
    # # 'deepseek-r1:7b': {
    # #     'model': "deepseek-r1:7b",
    # #     'provider': LLMProviders.OLLAMA
    # # },
    # # 'deepseek-r1:8b': {
    # #     'model': "deepseek-r1:7b",
    # #     'provider': LLMProviders.OLLAMA
    # # },
    # # 'deepseek-r1-14b': {
    # #     'model': "deepseek-r1-14b",
    # #     'provider': LLMProviders.OLLAMA
    # # },
    # # 'deepseek-r1-32b': {
    # #     'model': "deepseek-r1-32b",
    # #     'provider': LLMProviders.OLLAMA
    # # },
    'gpt-4o-abacus': {
        'model': 'OPENAI_GPT4O',
        'provider': LLMProviders.ABACUS,
    },
    'claude-sonnet-4-bedrock': {
        'model': 'us.anthropic.claude-sonnet-4-20250514-v1',
        'provider': LLMProviders.AWS_BEDROCK,
        'env': {'region': 'us-east-2'}  # default region
    },
    'qwen3': {
        'model': 'qwen3:32b',
        'provider': LLMProviders.OLLAMA_CHAT,
    },
    'qwen3-thinking': {                      # “thinking-mode” variant
        'model': 'qwen3:32b',
        'provider': LLMProviders.OLLAMA_CHAT,
        'thinking': True,                   # flag consumed later
    },
    'claude-sonnet-4-20250514-thinking': {
        'model': "claude-sonnet-4-20250514",
        'provider': LLMProviders.CLAUDE,
        'thinking_budget_tokens': 2000,
    },
    'claude-opus-4-20250514-thinking': {
        'model': "claude-opus-4-20250514",
        'provider': LLMProviders.CLAUDE,
        'thinking_budget_tokens': 2000,
    },
    'claude-sonnet-4-5-20250929-thinking': {
        'model': "claude-sonnet-4-5-20250929-thinking",
        'provider': LLMProviders.CLAUDE,
        'thinking_budget_tokens': 2000,
    },
    'o3-high': {
        'model': 'o3',  # deployment name
        'provider': LLMProviders.AZURE,
        'api_version': '2025-03-01-preview',
        'reasoning_effort': 'high',
    },
    'claude-v4-sonnet-abacus': {
        'model': 'CLAUDE_V4_SONNET',
        'provider': LLMProviders.ABACUS,
    },
    'claude-v4-opus-abacus': {
        'model': 'CLAUDE_V4_OPUS',
        'provider': LLMProviders.ABACUS,
    },
    'gemini-2-5-pro-abacus': {
        'model': 'GEMINI_2_5_PRO',
        'provider': LLMProviders.ABACUS,
    },
    'gemini-2.5-pro-native': {
        'model': 'gemini-2.5-pro',
        'provider': LLMProviders.GEMINI,
    },
    'grok-4-or': {
        'model': 'x-ai/grok-4',
        'provider': LLMProviders.OPENROUTER,
    },
    'devstral-med-or': {
        'model': 'mistralai/devstral-medium',
        'provider': LLMProviders.OPENROUTER,
    },
    'grok-4-fast-or': {
        'model': 'x-ai/grok-4-fast',
        'provider': LLMProviders.OPENROUTER,
    },
    'grok-code-fast-1-or': {
        'model': 'x-ai/grok-code-fast-1',
        'provider': LLMProviders.OPENROUTER,
    },
    'qwen3-coder-480b-or': {
        'model': 'qwen/qwen3-coder',
        'provider': LLMProviders.OPENROUTER,
    },
    'gpt-oss:120b': {
        'model': 'gpt-oss:120b',
        'provider': LLMProviders.OLLAMA,
    },
    'gpt-5-minimal': {
        'model': 'gpt-5',  # deployment name
        'provider': LLMProviders.AZURE,
        'api_version': '2025-03-01-preview',
        'reasoning_effort': 'minimal',
    },
    'gpt-5-low': {
        'model': 'gpt-5',  # deployment name
        'provider': LLMProviders.AZURE,
        'api_version': '2025-03-01-preview',
        'reasoning_effort': 'low',
    },
    'gpt-5-high': {
        'model': 'gpt-5',  # deployment name
        'provider': LLMProviders.AZURE,
        'api_version': '2025-03-01-preview',
        'reasoning_effort': 'high',
    },
    'qwen3-coder': {
        'model': 'qwen3-coder:30b',
        'provider': LLMProviders.OLLAMA,
    },
    'grok-4-abacus': {
        'model': 'XAI_GROK_4',
        'provider': LLMProviders.ABACUS,
    },
}

audio_base_models_path = {
    'whisper-large-v3': "openai/whisper-large-v3",
    'whisper-large-v3-most-recent': "openai/whisper-large-v3",
    'whisper-large-v2': "openai/whisper-large-v2"
}

audio_peft_models_path = {
    'whisper-large-v3': "shray98/peft_tokenized-openai-whisper-large-v3",
    # 'whisper-large-v3': "shray98/peft_gpcam-openai-whisper-large-v3",
    'whisper-large-v3-most-recent': "shray98/peft-20241126_145531-openai-whisper-large-v3",
    'whisper-large-v2': "shray98/peft_tokenized-openai-whisper-large-v2"
}

# Model configurations for different environments/purposes
model_configurations = {
    "local": {
        "text": {
            "default": "qwen2",
            "classifier": "qwen2", #"qwen2", #"gpt-4o",
            "operator": "qwen2.5-coder",  #"qwen2.5-coder",
            "analysis": "qwen2.5-coder",
            "refinement": "gpt-4o",
            "verifier": "gpt-4o", #"claude-3.5-sonnet"
            "pv_evaluator": "athene-v2"
        },
        "audio": {
            "default": "whisper-large-v3-most-recent",
            "transcription": "whisper-large-v3-most-recent"
        },
        "finetuned": {
            "text": False,
            "audio": True
        }
    },
    "abacus": {
        "text": {
            "default": "gpt-4o-abacus",
            "classifier": "gpt-4o-abacus",
            "operator": "gpt-4o-abacus",
            "analysis": "gpt-4o-abacus",
            "refinement": "gpt-4o-abacus",
            "verifier": "gpt-4o-abacus"
        },
        "audio": {
            "default": "whisper-large-v3-most-recent",
            "transcription": "whisper-large-v3-most-recent"
        },
        "finetuned": {
            "text": False,
            "audio": True
        }
    },
    "azure": {
        "text": {
            "default": "gpt-4o",
            "classifier": "gpt-4o",
            "operator": "gpt-4o",
            "analysis": "gpt-4o",
            "refinement": "gpt-4o",
            "verifier": "gpt-4o",
            "pv_evaluator": "gpt-5-low"
        },
        "audio": {
            "default": "whisper-large-v3-most-recent",
            "transcription": "whisper-large-v3-most-recent"
        },
        "finetuned": {
            "text": False,
            "audio": True
        }
    },
    "claude": {
        "text": {
            "default": "claude-4.5-sonnet",
            "classifier": "claude-4.5-sonnet",
            "operator": "claude-4.5-sonnet",
            "analysis": "claude-4.5-sonnet",
            "refinement": "claude-4.5-sonnet"
        },
        "audio": {
            "default": "whisper-large-v3-most-recent",
            "transcription": "whisper-large-v3-most-recent"
        },
        "finetuned": {
            "text": False,
            "audio": True
        }
    },
    "claude-bedrock": {
        "text": {
            "default": "claude-sonnet-4-bedrock",
            "classifier": "claude-sonnet-4-bedrock",
            "operator": "claude-sonnet-4-bedrock",
            "analysis": "claude-sonnet-4-bedrock",
            "refinement": "claude-sonnet-4-bedrock"
        },
        "audio": {
            "default": "whisper-large-v3-most-recent",
            "transcription": "whisper-large-v3-most-recent"
        },
        "finetuned": {
            "text": False,
            "audio": True
        }
    },
}

# Current active configuration
ACTIVE_CONFIG = "claude"  # Change this to "azure" to use Azure models


# ------------------------------------------------------------------ #
#  Exponential-backoff wrapper (max 5 tries, 1-60 s wait)
# ------------------------------------------------------------------ #
def _call_with_retry(func, *args, **kwargs):
    def _log_before_sleep(retry_state):
        exc = None
        if retry_state.outcome:
            try:
                exc = retry_state.outcome.exception()
            except Exception:
                pass

        fn_name = getattr(func, "__qualname__", getattr(func, "__name__", str(func)))
        next_sleep = getattr(getattr(retry_state, "next_action", None), "sleep", None)

        err_name = type(exc).__name__ if exc else "UnknownError"
        print(f"[_call_with_retry] retry {retry_state.attempt_number} for {fn_name} "
              f"next_sleep={next_sleep} error={err_name}: {exc}")

    @retry(
        wait=wait_exponential(multiplier=2,  min=1, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
        before_sleep=_log_before_sleep,
    )
    def _inner():
        return func(*args, **kwargs)
    return _inner()

def map_to_one_hot(input_class):
    # Define the mapping dictionary
    mapping = {
        "Op": "[1, 0, 0, 0, 0]",
        "Ana": "[0, 1, 0, 0, 0]",
        "Notebook": "[0, 0, 1, 0, 0]",
        "gpcam": "[0, 0, 0, 1, 0]",
        "xicam": "[0, 0, 0, 0, 1]"
    }

    # Return the mapped value or raise an error if not found
    return mapping.get(input_class)


def map_to_id(input_class):
    # Define the mapping dictionary
    mapping = {
        "Op": 0,
        "Ana": 1,
        "Notebook": 2,
        "gpcam": 3,
        "xicam": 4
    }

    # Return the mapped value or raise an error if not found
    return mapping.get(input_class)


def load_json_schema(beamline, cog):
    try:
        # Construct the module path
        module_path = f"src.hal_beam_com.beamline_prompts.{beamline}.base_prompts"

        # Dynamically import the module
        base_prompts_module = importlib.import_module(module_path)

        json_schema_name = 'json_schema_' + base_prompts[cog]

        # Access the desired prompt variable
        json_schema = getattr(base_prompts_module, json_schema_name)


    except ModuleNotFoundError:
        raise ValueError(f"No prompts found for beamline: {beamline}")
    except AttributeError:
        raise ValueError(f"No schema named '{json_schema_name}' in {beamline}.prompts")

    return json_schema


def indent_preserving_newlines(text, base_indent=0):
    """
    Indent a block of text while preserving newlines and bullet points for Jinja2.

    Args:
        text (str): The text to indent.
        base_indent (int): The number of levels to indent the text.
    """

    indent = "    " * base_indent  # 4 spaces per level
    lines = text.split("\n")

    # First line doesn't need indentation as it comes after the bullet point
    result = [lines[0]]

    # Subsequent lines should align with the start of the first line's text
    if len(lines) > 1:
        # Add 2 spaces after the bullet point's indent
        continuation_indent = indent + "  "
        result.extend(f"{continuation_indent}{line}" if line.strip() else ""
                      for line in lines[1:])

    return "\n".join(result)


def create_dynamic_op_prompt(beamline, user_id):
    # Setup Jinja2 environment
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cogs", "prompt_templates",
                                "jinja_templates")
    env = Environment(loader=FileSystemLoader(template_dir))
    env.filters['dynamic_indent'] = indent_preserving_newlines
    template = env.get_template("op_prompt.j2")

    # Load JSON data
    command_examples_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "beamline_prompts", beamline, "command_examples.json")
    with open(command_examples_path, "r") as file:
        command_json = json.load(file)

    user_command_json = None
    user_command_examples_path = os.path.join(get_command_dir(), beamline, f"{user_id}_command_examples.json")
    if os.path.exists(user_command_examples_path):
        with open(user_command_examples_path, "r") as file:
            user_command_json = json.load(file)

    # Load complex flow sections if available
    complex_flow = ""
    complex_flow_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "beamline_prompts", beamline, "complex_flow.txt")

    if os.path.exists(complex_flow_path):
        with open(complex_flow_path, "r", encoding="utf-8") as file:
            complex_flow = file.read()

    # Load beamline-specific prompt sections if available
    beamline_specifics = []
    beamline_specifics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "beamline_prompts", beamline, "beamline_specifics")

    if os.path.exists(beamline_specifics_path):
        # Check if the beamline_specifics directory is empty
        if os.listdir(beamline_specifics_path):
            # Loop through all files in the directory and append their contents
            for filename in sorted(os.listdir(beamline_specifics_path)):
                file_path = os.path.join(beamline_specifics_path, filename)
                if os.path.isfile(file_path):
                    with open(file_path, "r", encoding="utf-8") as file:
                        user_specifics = file.read()
                        beamline_specifics.append({'filename': filename, 'file_content': user_specifics})
    
    # Separate default and user commands
    default_commands = [item for item in command_json if item.get('cog') == 'Op' and item.get('default', True)]

    # Organize default commands by class
    default_command_classes = defaultdict(list)
    for item in default_commands:
        if not item.get('function'):
            continue
        class_name = item.get("class", "Miscellaneous Commands")
        default_command_classes[class_name].append(item)

    user_command_classes = defaultdict(list)
    if user_command_json:
        user_commands = [item for item in user_command_json if item.get('cog') == 'Op']
        for item in user_commands:
            if not item.get('function'):
                continue
            class_name = item.get("class", "Miscellaneous Commands")
            user_command_classes[class_name].append(item)

    # Render template with both default and user commands
    return template.render(classes=default_command_classes, user_commands=user_command_classes,
                           complex_flow=complex_flow, beamline_specifics=beamline_specifics)



def create_dynamic_ana_prompt(beamline, user_id):
    # Setup Jinja2 environment
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cogs", "prompt_templates",
                                "jinja_templates")
    env = Environment(loader=FileSystemLoader(template_dir))
    env.filters['dynamic_indent'] = indent_preserving_newlines
    template = env.get_template("ana_prompt.j2")

    # Load JSON data
    examples_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "beamline_prompts", beamline, "command_examples.json")

    with open(examples_path, "r") as file:
        data = json.load(file)

    # Separate default and user commands
    user_commands = [item for item in data if
                     item.get('cog') == 'Ana' and not item.get('default', True) and item.get('output') != ""]
    default_commands = [item for item in data if item.get('cog') == 'Ana' and item.get('default', True)]

    # Organize default commands by class
    classes = defaultdict(list)
    for item in default_commands:
        if not item.get('function'):
            continue
        class_name = item.get("class", "Miscellaneous Commands")
        classes[class_name].append(item)

    # Render template with both default and user commands
    return template.render(classes=classes, user_commands=user_commands)


def load_examples_json(cog, beamline, include_context_functions=False, include_only_context_functions=False,
                       system_prompt_type=None, testing=False):
    file_path = os.path.dirname(os.path.abspath(__file__))
    examples_path = os.path.join(file_path, "beamline_prompts", beamline, "command_examples.json")

    with open(examples_path, "r") as file:
        data = json.load(file)

    examples_string = "\n\nExamples:\n"

    match cog:
        # TODO: I think all of these have duplicate code that can be simplified
        case CogType.CLASSIFIER:
            example_counter = 1
            for i, item in enumerate(data, start=1):
                if not 'cog' in item:
                    print(
                        f"Skipping item {i} containing {item} for classifier dynamic prompt because it does not have a 'cog' field")
                    continue

                # Skip non-default items if context functions are not included
                if not include_context_functions and not item.get('default', False):
                    continue

                # TODO: Let's put a max on the number of examples of cogs for the classifier agent

                # Base example
                for example_input in item.get("example_inputs", []):
                    cog = item['cog']
                    if testing == True and system_prompt_type == SystemPromptType.LIST_OUTPUT:
                        cog = map_to_one_hot(item['cog'])
                    elif testing == True and system_prompt_type == SystemPromptType.ID_OUTPUT:
                        cog = map_to_id(item['cog'])

                    examples_string += f"Example {example_counter}:\n"
                    examples_string += f"User Prompt: {example_input}\n"
                    examples_string += f"Your Output: {cog}\n\n"
                    example_counter += 1
                    break

                # Add 'usage' example
                for usage in item.get("usage", []):
                    cog = item['cog']
                    if testing == True and system_prompt_type == SystemPromptType.LIST_OUTPUT:
                        cog = map_to_one_hot(item['cog'])
                    elif testing == True and system_prompt_type == SystemPromptType.ID_OUTPUT:
                        cog = map_to_id(item['cog'])

                    examples_string += f"Example {example_counter}:\n"
                    examples_string += f"User Prompt: {usage['input']}\n"
                    examples_string += f"Your Output: {cog}\n\n"
                    example_counter += 1
                    break

        # TODO: Currently not used, test and throw away?
        case CogType.OP:
            example_counter = 1
            for item in data:
                # TODO: Make these enums (only possible if everything is consistent and we interpret them as enums)
                if item['cog'] == 'Op' and item['output'] != "":
                    # Exclude non-default examples if not including context functions
                    if not include_context_functions and not item['default']:
                        continue

                    # Exclude default examples if including only context functions
                    if include_only_context_functions and item['default']:
                        continue

                    for example_input in item.get("example_inputs", []):
                        examples_string += f"Example {example_counter}:\n"
                        examples_string += f"User Prompt: {example_input}\n"
                        examples_string += f"Your Output: {item['output']}\n\n"
                        example_counter += 1

        case CogType.ANA:
            example_counter = 1
            for item in data:
                # TODO: Make these enums (only possible if everything is consistent and we interpret them as enums)
                if item['cog'] == 'Ana' and item['output'] != "":
                    # Exclude non-default examples if not including context functions
                    if not include_context_functions and not item['default']:
                        continue

                    for example_input in item.get("example_inputs", []):
                        examples_string += f"Example {example_counter}:\n"
                        examples_string += f"User Prompt: {example_input}\n"
                        examples_string += f"Your Output: {item['output']}\n\n"
                        example_counter += 1

    # Remove the trailing newline
    examples_string = examples_string.strip()

    return examples_string


def load_system_prompt(beamline, cog, data, system_prompt_path, system_prompt_type=None, testing=False, history = ""):
    if system_prompt_path:
        try:
            # Try UTF-8 first
            with open(system_prompt_path, 'r', encoding='utf-8') as f:
                system_prompt = f.read()
        except UnicodeDecodeError:
            # If UTF-8 fails, try other encodings
            try:
                with open(system_prompt_path, 'r', encoding='latin-1') as f:
                    system_prompt = f.read()
            except Exception as e:
                print(f"Error reading file: {e}")
                raise
    else:
        system_prompt = create_dynamic_system_prompt(cog=cog, data = data, beamline=beamline,
                                                    include_context_functions=data[0]['include_context_functions'],
                                                system_prompt_type=system_prompt_type, testing=testing, history = history)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    system_prompt
                )
            ),
            HumanMessagePromptTemplate.from_template("{text}"),
        ]
    )
    return prompt_template


def create_dynamic_system_prompt(cog, data, beamline, include_context_functions=False, system_prompt_path=None,
                                 system_prompt_type=None, testing=False, history = ""):
    system_prompt = ""

    if testing:
        print("IN TESTING")
        try:
            # Construct the module path
            module_path = f"src.hal_beam_com.beamline_prompts.{beamline}.testing_base_prompts"

            # Dynamically import the module
            base_prompts_module = importlib.import_module(module_path)

            base_prompt_name = base_prompts_testing[(cog, system_prompt_type)]

            # Access the desired prompt variable
            base_prompt = getattr(base_prompts_module, base_prompt_name)
            system_prompt = base_prompt

            if cog in (CogType.CLASSIFIER, CogType.OP, CogType.ANA):
                examples = load_examples_json(cog, beamline, include_context_functions=include_context_functions,
                                              system_prompt_type=system_prompt_type, testing=testing)
                system_prompt += examples

        except ModuleNotFoundError:
            raise ValueError(f"No prompts found for beamline: {beamline}")
        except AttributeError:
            raise ValueError(f"No prompt named '{base_prompt_name}' in {beamline}.prompts")

    else:
        if system_prompt_path is None:
            try:
                # Construct the module path
                module_path = f"src.hal_beam_com.beamline_prompts.{beamline}.base_prompts"

                # Dynamically import the module
                base_prompts_module = importlib.import_module(module_path)

                base_prompt_name = base_prompts[cog]

                # Access the desired prompt variable
                base_prompt = getattr(base_prompts_module, base_prompt_name)

            except ModuleNotFoundError:
                raise ValueError(f"No prompts found for beamline: {beamline}")
            except AttributeError:
                raise ValueError(f"No prompt named '{base_prompt_name}' in {beamline}.prompts")

            examples = ""

            # Format the relevant examples based on the cog type
            match cog:
                case CogType.CLASSIFIER:
                    base_prompt = base_prompt.format(history = history)
                    examples = load_examples_json(cog, beamline, include_context_functions=include_context_functions)

                case CogType.ANA:
                    # Separate handling for CogType.ANA if needed
                    base_prompt = base_prompt.format(history = history)
                    examples = create_dynamic_ana_prompt(beamline, data[0]["user_id"])

                case CogType.OP:
                    # Separate handling for CogType.OP if needed
                    base_prompt = base_prompt.format(history = history)
                    examples = create_dynamic_op_prompt(beamline, data[0]["user_id"])
                    

                case CogType.VERIFIER:
                    base_prompt = base_prompt.format(text_input = data[0]['text_input'], operator_cog_output = data[0]['operator_cog_output'],
                                                     operator_cog_system_prompt = data[0]['operator_cog_system_prompt'])
                    return base_prompt

            system_prompt = base_prompt + examples

    # TODO: Add ending prompt?

    return system_prompt


@lru_cache(maxsize=None)
def load_model(
        base_model: str,
        *,
        azure_api_key: str | None = None,
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
) -> any:
    """
    Load a text model with caching.
    Optionally pass `azure_api_key`, `azure_endpoint`, and `azure_deployment` to override or provide per-model values.
    If omitted, the function looks for `<MODEL>_AZURE_API_KEY`, `<MODEL>_AZURE_API_BASE`, `<MODEL>_AZURE_DEPLOYMENT`, falling back to the generic `AZURE_*` variables.
    If no explicit mapping exists in `base_models_path`, the variable names are
    auto-generated from the model name (e.g. ‘gpt-4o’ → GPT_4O_AZURE_API_KEY).
    """
    model_cfg = base_models_path[base_model]
    if isinstance(model_cfg, (list, tuple)):      # backward-compat
        model_name, provider = model_cfg
        env_cfg = {}
    else:
        model_name  = model_cfg['model']
        provider    = model_cfg['provider']
        env_cfg     = model_cfg.get('env', {})

    match provider:
        case LLMProviders.OLLAMA:
            llm = Ollama(base_url="http://localhost:11434", model=model_name, temperature=0)
        case LLMProviders.OLLAMA_CHAT:
            llm = ChatOllama(
                base_url   = "http://localhost:11434",
                model      = model_name,
                temperature= 0,
                reasoning  = model_cfg.get("thinking", False)
            )
        case LLMProviders.HUGGING_FACE:
            llm = HuggingFacePipeline.from_model_id(
                model_id=model_name,
                task="text-generation",
                model_kwargs={'trust_remote_code': True, 'device_map': 'auto'},
                pipeline_kwargs=dict(max_new_tokens=512,
                                     do_sample=False,
                                     repitition_penalty=1.03)
            )
        case LLMProviders.AZURE:
            # -----------------------------------------------------------------
            # 1) determine the environment-variable *names*
            # -----------------------------------------------------------------
            def _sanitize(name: str) -> str:
                # upper-case and replace non-alnum with “_”
                return re.sub(r"[^A-Z0-9]", "_", name.upper())

            safe_prefix = _sanitize(model_name)
            api_key_env   = env_cfg.get("api_key",   f"{safe_prefix}_AZURE_API_KEY")
            endpoint_env  = env_cfg.get("endpoint",  f"{safe_prefix}_AZURE_API_BASE")
            deploy_env    = env_cfg.get("deployment",f"{safe_prefix}_AZURE_DEPLOYMENT")

            # -----------------------------------------------------------------
            # 2) resolve the *values* (explicit kwargs → per-model var →
            #    standard Azure/OpenAI vars)
            # -----------------------------------------------------------------
            api_key = (
                azure_api_key
                or os.getenv(api_key_env)
                or os.getenv("AZURE_OPENAI_API_KEY")
                or os.getenv("AZURE_API_KEY")
                or os.getenv("OPENAI_API_KEY")
            )

            endpoint = (
                azure_endpoint
                or os.getenv(endpoint_env)
                or os.getenv("AZURE_OPENAI_ENDPOINT")
                or os.getenv("AZURE_OPENAI_API_BASE")
                or os.getenv("AZURE_API_BASE")
                or os.getenv("OPENAI_API_BASE")
            )

            deployment = (
                azure_deployment
                or os.getenv(deploy_env)
                or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
                or os.getenv("AZURE_DEPLOYMENT")
            )

            # prepend scheme if missing
            if endpoint and not endpoint.startswith(("http://", "https://")):
                endpoint = f"https://{endpoint.lstrip('/')}"

            # -----------------------------------------------------------------
            # 3) sanity-check
            # -----------------------------------------------------------------
            missing = []
            if not api_key:
                missing.append("API key")
            if not endpoint:
                missing.append("endpoint")
            if not deployment:
                missing.append("deployment name")
            if missing:
                raise ValueError(
                    f"Azure credentials for model '{base_model}' are incomplete "
                    f"({', '.join(missing)} missing). "
                    f"Searched: {api_key_env}, {endpoint_env}, {deploy_env}, "
                    "AZURE_OPENAI_*, AZURE_*, OPENAI_* or explicit kwargs."
                )

            # Pass `reasoning_effort` only when it is explicitly configured.
            reasoning_effort = model_cfg.get("reasoning_effort")

            # -----------------------------------------------------------------
            # 4) create the client
            # -----------------------------------------------------------------
            az_kwargs = dict(
                api_key          = api_key,
                azure_endpoint   = endpoint,
                azure_deployment = deployment,
                api_version      = model_cfg.get("api_version", "2023-05-15"),
                temperature      = 1 if reasoning_effort else 0,
            )

            if reasoning_effort is not None:
                az_kwargs["reasoning_effort"] = reasoning_effort

            llm = AzureChatOpenAI(**az_kwargs)
        case LLMProviders.CLAUDE:
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            llm = Anthropic(api_key=api_key)
            llm.model_name = model_name  # Store the model name on the client instance
            # store optional thinking budget on the client instance
            thinking_budget = model_cfg.get('thinking_budget_tokens')
            if thinking_budget:
                llm.thinking_budget_tokens = thinking_budget
        case LLMProviders.ABACUS:
            api_key = os.getenv("ABACUS_API_KEY")
            if not api_key:
                raise ValueError("ABACUS_API_KEY environment variable not set")
            llm = abacusai.ApiClient(api_key=api_key)     # NEW SDK
            llm.model_name = model_name                   # keep for caller
        case LLMProviders.AWS_BEDROCK:
            if boto3 is None:
                raise ImportError("boto3 is required for AWS_BEDROCK provider")

            # ------------------------------------------------------------------
            # Resolve credentials / region
            # ------------------------------------------------------------------
            region      = (env_cfg.get("region") or
                           os.getenv("AWS_BEDROCK_REGION") or
                           os.getenv("AWS_REGION") or "us-east-2")
            acc_id      = (env_cfg.get("account_id") or
                           os.getenv("AWS_ACCOUNT_ID") or
                           os.getenv("AWS_BEDROCK_ACCOUNT_ID"))
            access_key  = os.getenv("AWS_ACCESS_KEY_ID")
            secret_key  = os.getenv("AWS_SECRET_ACCESS_KEY")

            # ------------------------------------------------------------------
            # Build / accept full ARN
            # ------------------------------------------------------------------
            if model_name.startswith("arn:"):
                model_id = model_name
            else:
                if not acc_id:
                    raise ValueError(
                        "AWS_ACCOUNT_ID environment variable missing – "
                        "required to build model ARN"
                    )
                model_id = (
                    f"arn:aws:bedrock:{region}:{acc_id}:"
                    f"inference-profile/{model_name}:0"
                )

            client = boto3.client(
                "bedrock-runtime",
                region_name=region,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
            )

            llm = BedrockLLM(client, model_id)
        case LLMProviders.GEMINI:
            if genai is None:
                raise ImportError("google-genai is required for GEMINI provider")
            # The client reads GEMINI_API_KEY from the environment by default.
            client = genai.Client()
            llm = GeminiLLM(client, model_name)
        case LLMProviders.OPENROUTER:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set")
            llm = ChatOpenAI(
                openai_api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                model=model_name,
                temperature=0,
            )
        case _:
            raise ValueError(f"Unknown model provider: {provider}")

    print(f"Loaded model '{base_model}'")

    return llm


@lru_cache(maxsize=None)
def load_whisper_model(model_name, finetuned=False):
    """
    Load a whisper model with caching
    
    Args:
        model_name (str): Name of the whisper model to load
        finetuned (bool): Whether to load the finetuned version
        
    Returns:
        tuple: (model, processor) The loaded model and its processor
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    if model_name.startswith('whisper'):
        model_id = audio_base_models_path[model_name]
    else:
        entry = base_models_path[model_name]
        model_id = entry[0] if isinstance(entry, (list, tuple)) else entry['model']

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    ).to(device)

    if finetuned:
        model = PeftModel.from_pretrained(model, audio_peft_models_path[model_name])

    processor = AutoProcessor.from_pretrained(model_id)

    print(f"Loaded whisper model '{model_name}'")

    return model, processor


def prompt_to_string(prompt_template, cog, save=False):
    try:
        # If prompt_template is bytes, decode it first
        if isinstance(prompt_template, bytes):
            prompt_template = prompt_template.decode('utf-8')

        # Format the prompt template
        prompt_as_string = prompt_template.format(
            text=""
        )

        # Split and join lines
        lines = prompt_as_string.splitlines()
        prompt_as_string = "\n".join(lines[:-1])

        if save:
            # Create directory for final prompts if it doesn't exist
            final_prompts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "cogs", "final_prompts")
            os.makedirs(final_prompts_dir, exist_ok=True)

            # Save the prompt file in the final_prompts directory
            file_path = os.path.join(final_prompts_dir, f"final_prompt_{cog.value}.txt")

            # Ensure UTF-8 encoding is used for writing files
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(prompt_as_string)

        return prompt_as_string

    except UnicodeDecodeError:
        # Try with different encodings if utf-8 fails
        try:
            if isinstance(prompt_template, bytes):
                prompt_template = prompt_template.decode('latin-1')
                return prompt_to_string(prompt_template, cog, save)
        except:
            print("Error: Could not decode prompt template with UTF-8 or Latin-1 encoding")
            raise

def find_first_match(llm_output: str, allowed_values: list) -> str:
    text_words = llm_output.split()
    for word in text_words:
        if word in allowed_values:
            return word
    return "MISSED"

def find_last_match(llm_output: str, allowed_values: list) -> str:
    # Method 1: Using split and reversed
    return next((word for word in reversed(llm_output.split())
                 if word in allowed_values), "MISSED")


def process_llm_output(llm_output, prompt_type):
    next_cog = None

    if prompt_type == SystemPromptType.LIST_OUTPUT:

        lines = llm_output.strip().split('\n')
        # print(lines)
        for i, element in enumerate(lines):
            try:
                # Try to safely evaluate the string to a Python object
                parsed_element = ast.literal_eval(element)
                # Check if the parsed element is a list
                if isinstance(parsed_element, list):
                    print(f"Found list: {parsed_element}")
                    break  # Exit the loop once the list is found
            except (ValueError, SyntaxError):
                # If evaluation fails, continue to the next element
                continue

        try:
            confidence_values = ast.literal_eval(lines[i].strip())
        except:
            next_cog = "MISSED"
            return next_cog

        next_id = np.argmax(confidence_values)

        next_cog = id_to_cog[next_id]

    elif prompt_type == SystemPromptType.ONE_WORD_OUTPUT:

        # Using the last word in the LLM output that corresponds to a cog as the prediction
        return find_last_match(llm_output, id_to_cog.values())

    elif prompt_type == SystemPromptType.ID_OUTPUT:

        cog_id = find_id_in_string(llm_output)

        try:
            next_cog = id_to_cog[cog_id]
        except:
            next_cog = "MISSED"
            return next_cog

    return next_cog


def find_id_in_string(input_string):
    # Using regular expression to find the first number in the string
    match = re.search(r'\d+', input_string)
    return int(match.group()) if match else None


def wait_for_response(queue, data):
    # queue.publish(data)

    data = queue.get()

    if data[0]['bl_conf'] == 0:
        print("Did not receive confirmation from Beamline")

    if data[0]['bl_conf'] == 1:
        print("Received confirmation from Beamline")

    return data


def tab3_response(queue, data):
    data = queue.get()

    if data[0]['bl_tab3_conf'] == 0:
        print("Did not receive confirmation from Beamline")

    if data[0]['bl_tab3_conf'] == 1:
        print("Received confirmation from Beamline")

    return data


def save_audio_error(audio, text, audio_error_count):
    # Define base error directory
    error_base_dir = "./errors/voice_cog/"
    os.makedirs(error_base_dir, exist_ok=True)

    # Define and create subdirectories for audio and text
    audio_dir = os.path.join(error_base_dir, "audio")
    text_dir = os.path.join(error_base_dir, "text")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)

    # Construct file paths for audio and text
    audio_file_path = os.path.join(audio_dir, f"error_{audio_error_count}.wav")
    text_file_path = os.path.join(text_dir, f"error_{audio_error_count}.txt")

    # Save audio to file
    audio.export(audio_file_path, format="wav")

    # Save text to file
    with open(text_file_path, "w") as text_file:
        text_file.write(text)

    audio_error_count += 1


def set_data_fields(rdb_data, cog, data, debug_printing=False):
    for key, value in rdb_data.items():
        if debug_printing:
            print(key, value)
        data[0][f"{cog.value}_cog_{key}"] = value

    return data


def prepare_example_for_json(data, selected_cog):
    if isinstance(data, dict):
        data["cog"] = selected_cog  # Add the "cog" field with the value of selected_cog
        data["default"] = False  # Add the "default" field set to False
    return data


def get_data_field(rdb_data, cog, key):
    if not rdb_data:
        return None
    return rdb_data[0][f"{cog.value}_cog_{key}"]


def execute_llm_call(llm, prompt_template, user_prompt, strip_markdown=False, history = ""):
    """
    Executes an LLM call with standardized provider handling.
    
    Args:
        llm: The loaded language model
        prompt_template: The formatted prompt template
        user_prompt: The user's input text
        strip_markdown: Whether to strip markdown formatting from output
        
    Returns:
        str: The processed LLM output
    """
    match llm:
        case AzureChatOpenAI():
            chain = prompt_template | llm
            response = _call_with_retry(chain.invoke, {'text': user_prompt})
            output = response.content
        case Anthropic():
            messages = prompt_template.format_messages(text=user_prompt)
            system_message = messages[0].content
            human_message = messages[1].content

            reasoning_enabled = getattr(llm, "thinking_budget_tokens", False)

            # build request parameters
            params = dict(
                model       = llm.model_name,
                system      = system_message,
                messages    = [{"role": "user", "content": human_message}],
                temperature = 1 if reasoning_enabled else 0,
                max_tokens  = 3048 if reasoning_enabled else 1024 # allow long answers, needed for long reasoning window
            )

            # enable Claude-4 “thinking” if budget configured
            budget = getattr(llm, "thinking_budget_tokens", None)
            if budget:
                params["thinking"] = {"type": "enabled",
                                      "budget_tokens": budget}

            response = _call_with_retry(llm.messages.create, **params)

            # collect only the real answer, discard thinking summaries
            if isinstance(response.content, (list, tuple)):
                output = "".join(
                    block.text for block in response.content
                    if getattr(block, "type", None) == "text"
                )
            else:   # fallback to old response format
                output = response.content[0].text
        case AbacusClient():
            messages       = prompt_template.format_messages(text=user_prompt)
            system_message = messages[0].content
            human_message  = messages[1].content

            response = _call_with_retry(
                llm.evaluate_prompt,
                prompt=human_message,
                system_message=system_message,
                llm_name=getattr(llm, "model_name", None),
                max_tokens=1024,
                temperature=0.0,
            )
            output = response.content                      # LlmResponse attr
        case BedrockLLM():
            # 1) extract prompt parts
            msgs            = prompt_template.format_messages(text=user_prompt)
            system_message  = msgs[0].content
            human_message   = msgs[1].content

            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens"      : 1024,
                "temperature"     : 0.0,
                "system"          : system_message,          # Claude wants this here
                "messages": [
                    {"role": "user", "content": human_message},
                ],
            }

            try:
                resp = _call_with_retry(
                    llm.client.invoke_model,
                    modelId     = llm.model_id,
                    body        = json.dumps(payload),
                    contentType = "application/json",
                    accept      = "application/json",
                )
                body = json.loads(resp["body"].read())
                if isinstance(body.get("content"), list):
                    output = "".join(part.get("text", "")
                                     for part in body["content"])
                else:                                   # safety-fallback
                    output = body.get("content") or body.get("completion", "")
            except ClientError as e:
                raise RuntimeError(f"Bedrock error: {e.response['Error']['Message']}")
        case GeminiLLM():
            messages = prompt_template.format_messages(text=user_prompt)
            system_message = messages[0].content
            human_message = messages[1].content
            if genai_types is not None:
                try:
                    config = genai_types.GenerateContentConfig(
                        system_instruction=system_message,
                        temperature=0.0,
                    )
                    response = _call_with_retry(
                        llm.client.models.generate_content,
                        model=llm.model_name,
                        contents=human_message,
                        config=config,
                    )
                except TypeError:
                    # Fallback: concatenate system + user messages when config/system_instruction isn't supported
                    response = _call_with_retry(
                        llm.client.models.generate_content,
                        model=llm.model_name,
                        contents=f"{system_message}\n\n{human_message}",
                    )
            else:
                # Fallback when google.genai.types isn't available
                response = _call_with_retry(
                    llm.client.models.generate_content,
                    model=llm.model_name,
                    contents=f"{system_message}\n\n{human_message}",
                )
            output = response.text
        case ChatOpenAI():
            chain = prompt_template | llm
            response = _call_with_retry(chain.invoke, {'text': user_prompt})
            output = response.content
        case ChatOllama():
            # build the message list from the prompt template
            messages = prompt_template.format_messages(text=user_prompt)
            response = _call_with_retry(llm.invoke, messages)   # reasoning handled by llm
            output = response.content
        case _:
            chain = prompt_template | llm.bind(skip_prompt=True)
            output = _call_with_retry(
                chain.invoke,
                {'history': history, 'text': user_prompt},
            )

    if strip_markdown:
        output = str(re.sub(r"```(?:python)?", "", output)).lstrip()

    return output


def execute_structured_llm_call(llm, prompt_template, user_prompt, json_schema):
    """
    Executes an LLM call for structured output with standardized provider handling.
    
    Args:
        llm: The loaded language model
        prompt_template: The formatted prompt template
        user_prompt: The user's input text
        json_schema: The JSON schema for structured output
        
    Returns:
        dict: The structured output
    """

    match llm:
        case AzureChatOpenAI():
            structured_llm = llm.with_structured_output(json_schema)
            chain = prompt_template | structured_llm
            output = chain.invoke({'text': user_prompt})

        case ChatOpenAI():
            structured_llm = llm.with_structured_output(json_schema)
            chain = prompt_template | structured_llm
            output = chain.invoke({'text': user_prompt})

        case Anthropic():
            messages = prompt_template.format_messages(text=user_prompt)
            system_message = messages[0].content
            human_message = messages[1].content
            response = llm.messages.create(
                model=llm.model_name,
                system=system_message + "\nOutput must conform to this JSON schema: " + str(json_schema),
                messages=[
                    {"role": "user", "content": human_message}
                ],
                temperature=0,
                max_tokens=1024
            )
            output = json.loads(response.content[0].text)

        case GeminiLLM():
            messages = prompt_template.format_messages(text=user_prompt)
            system_message = messages[0].content
            human_message = messages[1].content
            if genai_types is not None:
                try:
                    config = genai_types.GenerateContentConfig(
                        system_instruction=system_message,
                        response_mime_type="application/json",
                        temperature=0.0,
                    )
                    response = llm.client.models.generate_content(
                        model=llm.model_name,
                        contents=human_message,
                        config=config,
                    )
                except TypeError:
                    # Fallback when config/system_instruction not supported
                    response = llm.client.models.generate_content(
                        model=llm.model_name,
                        contents=f"{system_message}\n\n{human_message}\n\nReturn only valid JSON conforming to this schema:\n{json_schema}"
                    )
            else:
                response = llm.client.models.generate_content(
                    model=llm.model_name,
                    contents=f"{system_message}\n\n{human_message}\n\nReturn only valid JSON conforming to this schema:\n{json_schema}"
                )
            output = json.loads(response.text)

        case _:
            raise NotImplementedError
        
    return output


def convert_dict_to_string(input_data):
    if isinstance(input_data, dict):
        return str(input_data)  # Converts the dictionary to a string
    return input_data


def ensure_beamline_folder(base_folder, beamline):
    """
    Ensures the beamline folder exists within the base folder.
    Creates the folder if it does not exist.

    Args:
        base_folder (str): The root folder containing beamline folders.
        beamline (str): The name of the beamline folder to check or create.

    Returns:
        str: The path to the beamline folder.
    """
    beamline_folder = os.path.join(base_folder, beamline)
    os.makedirs(beamline_folder, exist_ok=True)

    # Check if beamline_specifics.txt exists, create default if not
    specifics_path = os.path.join(beamline_folder, "beamline_specifics.txt")
    if not os.path.exists(specifics_path):
        # We'll let the template provide the defaults instead of creating an empty file
        pass

    # Check if complex_flow.txt exists
    complex_flow_path = os.path.join(beamline_folder, "complex_flow.txt")
    if not os.path.exists(complex_flow_path):
        # We'll let the template provide the defaults instead of creating an empty file
        pass

    return beamline_folder


def append_to_command_examples(data):
    """
    Appends a prepared dictionary to the command_examples.json file
    in the appropriate beamline folder. Creates the folder and JSON file
    if they do not exist.

    Args:
        data (dict): The input dictionary with 'input' and 'output' fields.
        data_dir (str, optional): The directory to save the JSON files in. Defaults to None.

    Returns:
        None
    """
    # Base folder for beamline prompts
    data_dir = get_command_dir()
    user_id = data[0]["user_id"]
    beamline = data[0]['beamline_id']

    # beamline_folder = ""
    if data_dir is None:
        root_dir = "vision_hal/src/hal_beam_com/"
        # base_folder = "beamline_prompts"
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        base_folder = os.path.join(SCRIPT_DIR, "beamline_prompts")
        # Ensure the beamline folder exists
        beamline_folder = ensure_beamline_folder(base_folder, beamline)
        # Path to the command_examples.json file
        json_file_path = os.path.join(beamline_folder, "command_examples.json")
        display_output_path = os.path.join(root_dir, json_file_path)
    else:
        beamline_folder = os.path.join(data_dir, beamline)
        os.makedirs(beamline_folder, exist_ok=True)
        print(f"beamline_folder: {beamline_folder}")

        json_file_path = os.path.join(beamline_folder, f"{user_id}_command_examples.json")
        display_output_path = json_file_path

    try:
        prepared_data = data[0]['refinement_cog_output_dict']
    except (KeyError, IndexError, TypeError):
        data[0]['append_examples_output'] = 'Failed to add the example'

    prepared_data = data[0]['refinement_cog_output_dict']

    # Check if the JSON file exists
    if not os.path.exists(json_file_path):
        # Create the JSON file and add prepared_data
        with open(json_file_path, "w") as file:
            json.dump([prepared_data], file, indent=4)
        print(f"JSON file '{json_file_path}' created and data added.")

    else:
        # Append the prepared_data to the existing JSON file
        try:
            with open(json_file_path, "r") as file:
                existing_data = json.load(file)

            # Ensure the file contains a list
            if isinstance(existing_data, list):
                existing_data.append(prepared_data)
            else:
                print("Existing JSON data is not a list. Converting to list.")
                existing_data = [existing_data, prepared_data]

            # Write the updated data back to the file
            with open(json_file_path, "w") as file:
                json.dump(existing_data, file, indent=4)

            print(f"Data appended to '{json_file_path}'.")

            bold_start = "<b>"
            bold_end = "</b>"

            data[0]['append_examples_output'] = (
                f"Successfully added {bold_start}{data[0]['refinement_cog_output_dict']['function']}{bold_end} "
                f"to the {bold_start}{data[0]['selected_cog']} Cog{bold_end} at "
                f"{bold_start}{data[0]['beamline_id']} Beamline{bold_end}. Added on HAL here - {bold_start}{display_output_path}{bold_end}!!")

        except json.JSONDecodeError as e:
            print(f"Error reading JSON file: {e}")


def get_model_config(model_type, cog_type="default", default="gpt-4o"):
    """
    Get the configured model name for a specific type and cog.
    
    Args:
        model_type (str): Either "text" or "audio"
        cog_type (str): The specific cog type (default, classifier, operator, etc.)
        default (str): The default model to return if keys aren't found
        
    Returns:
        str: The configured model name, or default if not found
    """
    try:
        return model_configurations[ACTIVE_CONFIG][model_type][cog_type]
    except (KeyError, TypeError):
        return default


def get_finetuned_config(model_type):
    """
    Get whether finetuning should be used for a model type.
    
    Args:
        model_type (str): Either "text" or "audio"
        
    Returns:
        bool: Whether to use finetuned models
    """
    return model_configurations[ACTIVE_CONFIG]["finetuned"][model_type]


def load_testing_model(i, word, torch_dtype):
    model_id = "openai/whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    
    peft_model_id = f"shray98/peft-cumulative-testing-all-{str(i)}-openai-whisper-large-v3"
    model = PeftModel.from_pretrained(model, peft_model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    print(f"LOADING MODEL from {peft_model_id}")
    return model, processor

def parse_verifier_json(response: str) -> dict:
    """
    Extract and parse JSON from between <response> and </response> tags.

    Args:
        response (str): Raw response string from LLM

    Returns:
        dict: Parsed JSON object

    Raises:
        ValueError: If response tags not found or JSON is invalid
    """
    try:
        # Extract content between tags
        pattern = r'<response>(.*?)</response>'
        match = re.search(pattern, response, re.DOTALL)  # re.DOTALL allows matching across multiple lines

        if not match:
            raise ValueError("Response tags not found in LLM output")

        # Get the JSON string and parse it
        json_str = match.group(1).strip()
        result = json.loads(json_str)

        # Ensure all expected keys are present
        expected_keys = {
            "missing_info",
            "hallucinations",
            "assumptions",
            "is_response_justified",
            "concerns",
            "verification_summary"
        }

        if not all(key in result for key in expected_keys):
            missing_keys = expected_keys - result.keys()
            raise ValueError(f"Missing required keys in JSON: {missing_keys}")

        return result

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in response: {e}")
    except Exception as e:
        raise ValueError(f"Error parsing response: {e}")
