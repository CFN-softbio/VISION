import os
import re
from enum import Enum
import json
import ast
import numpy as np
import importlib
from anthropic import Anthropic
import torch
from jinja2 import Environment, FileSystemLoader
from collections import defaultdict
from functools import lru_cache

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.llms import Ollama
from langchain_openai import AzureChatOpenAI
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

class CogType(Enum):
    VOICE = "voice"
    CLASSIFIER = "classifier"
    OP = "operator"
    ANA = "analysis"
    REFINE = "refinement"
    GPCAM = "gpcam"
    XICAM = "xicam"


class LLMProviders(Enum):
    OLLAMA = "ollama"
    HUGGING_FACE = "hugging_face"
    AZURE = "azure"
    CLAUDE = "claude"


base_prompts = {

    CogType.CLASSIFIER: 'classifier_cog',
    CogType.OP: 'op_cog',
    CogType.REFINE: 'refinement_cog',
    CogType.ANA: 'analysis_cog'

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
    6: f"{CogType.REFINE.value}_cog_output"
}

id_to_cog = {
    0: "Op",
    1: "Ana",
    2: "Notebook",
    3: "gpcam",
    4: "xicam"
}

base_models_path = {
    'mistral': ["mistral", LLMProviders.OLLAMA],
    # 'Llama-3_1-Nemotron-51B-instruct': ["nvidia/Llama-3_1-Nemotron-51B-Instruct", LLMProviders.HUGGING_FACE],
    'qwen2.5-coder': ["qwen2.5-coder:32b", LLMProviders.OLLAMA],
    'claude-3.5-sonnet': ["claude-3-5-sonnet-20241022", LLMProviders.CLAUDE],
    'qwen2': ["qwen2", LLMProviders.OLLAMA],
    'qwen2.5': ["qwen2.5", LLMProviders.OLLAMA],
    'mistral-nemo': ["mistral-nemo", LLMProviders.OLLAMA],
    'gpt-4o': ["gpt-4o", LLMProviders.AZURE],
    'phi3.5': ["phi3.5", LLMProviders.OLLAMA],
    'phi3.5-fp16': ["phi3.5:3.8b-mini-instruct-fp16", LLMProviders.OLLAMA],
    'athene-v2': ["athene-v2", LLMProviders.OLLAMA],
    'athene-v2-agent': ["hf.co/lmstudio-community/Athene-V2-Agent-GGUF", LLMProviders.OLLAMA],
    'llama3.3': ["llama3.3", LLMProviders.OLLAMA],
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
            "classifier": "qwen2",
            "operator": "qwen2.5-coder",
            "analysis": "qwen2",
            "refinement": "gpt-4o"
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
            "refinement": "gpt-4o"
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
            "default": "claude-3.5-sonnet",
            "classifier": "claude-3.5-sonnet",
            "operator": "claude-3.5-sonnet",
            "analysis": "claude-3.5-sonnet",
            "refinement": "claude-3.5-sonnet"
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


def create_dynamic_op_prompt(beamline):
    # Setup Jinja2 environment
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cogs", "prompt_templates",
                                "jinja_templates")
    env = Environment(loader=FileSystemLoader(template_dir))
    env.filters['dynamic_indent'] = indent_preserving_newlines
    template = env.get_template("op_prompt.j2")

    # Load JSON data
    examples_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "beamline_prompts", beamline, "command_examples.json")

    with open(examples_path, "r") as file:
        data = json.load(file)

    # Separate default and user commands
    user_commands = [item for item in data if
                     item.get('cog') == 'Op' and not item.get('default', True) and item.get('output') != ""]
    default_commands = [item for item in data if item.get('cog') == 'Op' and item.get('default', True)]

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


def load_system_prompt(beamline, cog, data, system_prompt_path, system_prompt_type=None, testing=False):
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
        system_prompt = create_dynamic_system_prompt(cog=cog, beamline=beamline,
                                                     include_context_functions=data[0]['include_context_functions'],
                                                     system_prompt_type=system_prompt_type, testing=testing)
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


def create_dynamic_system_prompt(cog, beamline, include_context_functions=False, system_prompt_path=None,
                                 system_prompt_type=None, testing=False):
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
                case CogType.CLASSIFIER | CogType.ANA:
                    examples = load_examples_json(cog, beamline, include_context_functions=include_context_functions)

                case CogType.OP:
                    # Separate handling for CogType.OP if needed
                    examples = create_dynamic_op_prompt(beamline)

            system_prompt = base_prompt + examples

    # TODO: Add ending prompt?

    return system_prompt


@lru_cache(maxsize=None)
def load_model(base_model):
    """
    Load a text model with caching
    """
    base_model_path, provider = base_models_path[base_model]

    match provider:
        case LLMProviders.OLLAMA:
            llm = Ollama(base_url="http://localhost:11434", model=base_model_path, temperature=0)
        case LLMProviders.HUGGING_FACE:
            llm = HuggingFacePipeline.from_model_id(
                model_id=base_model_path,
                task="text-generation",
                model_kwargs={'trust_remote_code': True, 'device_map': 'auto'},
                pipeline_kwargs=dict(max_new_tokens=512,
                                     do_sample=False,
                                     repitition_penalty=1.03)
            )
        case LLMProviders.AZURE:
            llm = AzureChatOpenAI(
                api_key=os.environ.get('AZURE_API_KEY'),
                azure_endpoint=os.environ.get('AZURE_API_BASE'),
                azure_deployment=os.environ.get('AZURE_DEPLOYMENT'),
                openai_api_version="2023-05-15",
                temperature=0.0,
            )
        case LLMProviders.CLAUDE:
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            llm = Anthropic(api_key=api_key)
            llm.model_name = base_model_path  # Store the model name on the client instance
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

    model_id = (audio_base_models_path[model_name]
                if model_name.startswith('whisper')
                else base_models_path[model_name][0])

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
            # Ensure UTF-8 encoding is used for writing files
            with open("final_prompt_" + cog.value + ".txt", "w", encoding="utf-8") as file:
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

        allowed_values = id_to_cog.values()

        if any(word in llm_output for word in allowed_values):

            for word in allowed_values:
                if word in llm_output:
                    next_cog = word

                    return next_cog

        else:
            next_cog = "MISSED"

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
    return rdb_data[0][f"{cog.value}_cog_{key}"]


def execute_llm_call(llm, prompt_template, user_prompt, strip_markdown=False):
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
            response = chain.invoke({'text': user_prompt})
            output = response.content
        case Anthropic():
            messages = prompt_template.format_messages(text=user_prompt)
            system_message = messages[0].content
            human_message = messages[1].content
            response = llm.messages.create(
                model=llm.model_name,
                system=system_message,
                messages=[
                    {"role": "user", "content": human_message}
                ],
                temperature=0,
                max_tokens=1024
            )
            output = response.content[0].text
        case _:
            chain = prompt_template | llm.bind(skip_prompt=True)
            output = chain.invoke({'text': user_prompt})

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
    return beamline_folder


def append_to_command_examples(data):
    """
    Appends a prepared dictionary to the command_examples.json file
    in the appropriate beamline folder. Creates the folder and JSON file
    if they do not exist.

    Args:
        data (dict): The input dictionary with 'input' and 'output' fields.

    Returns:
        None
    """
    # Base folder for beamline prompts

    beamline = data[0]['beamline']
    root_dir = "vision_hal/src/hal_beam_com/"
    base_folder = "beamline_prompts"

    # Ensure the beamline folder exists
    beamline_folder = ensure_beamline_folder(base_folder, beamline)

    # Path to the command_examples.json file
    json_file_path = os.path.join(beamline_folder, "command_examples.json")

    display_output_path = os.path.join(root_dir, json_file_path)

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
                f"Successfully added {bold_start}{data[0]['refinement_cog_output_dict']['output']}{bold_end} "
                f"to the {bold_start}{data[0]['selected_cog']} Cog{bold_end} at "
                f"{bold_start}{data[0]['beamline_id']} Beamline{bold_end}. Added on HAL here - {bold_start}{display_output_path}{bold_end}!!")

        except json.JSONDecodeError as e:
            print(f"Error reading JSON file: {e}")


def get_model_config(model_type, cog_type="default"):
    """
    Get the configured model name for a specific type and cog.
    
    Args:
        model_type (str): Either "text" or "audio"
        cog_type (str): The specific cog type (default, classifier, operator, etc.)
        
    Returns:
        str: The configured model name
    """
    return model_configurations[ACTIVE_CONFIG][model_type][cog_type]


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
