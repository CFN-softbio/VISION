import os
from enum import Enum

from semantic_router.encoders import HuggingFaceEncoder, AzureOpenAIEncoder
from semantic_router import Route
from semantic_router.layer import RouteLayer
from openai import AzureOpenAI
from paperqa import Settings, ask

from src.hal_beam_com.chatbot_prompts import (
    question_llm, 
    toolselector_llm, 
    papaerqa_lite,
    paperqa_lite_beamline,
    toolselector_llm_ui

)

from src.hal_beam_com.naive_rag_utils import (

    RAGConfig,
    RAGSystem,

)

def call_azure_openai_gpt4(prompt, task, history):

    system_prompt = f"Consider the conversation history if necessary \n\n History: {history}"

    client = AzureOpenAI(
    api_key = os.environ.get('AZURE_API_KEY'),  
    api_version='2023-05-15',
    azure_endpoint = os.environ.get('AZURE_API_BASE'),
    azure_deployment = os.environ.get('AZURE_DEPLOYMENT')
    )

    if task == "question":
        print("SETTING QUESTION SYSTEM PROMPT")
        system_prompt += question_llm

    if task == "toolselector":
        print("SETTING TOOL SELECTOR SYSTEM PROMPT")
        system_prompt = toolselector_llm

    if task == "toolselector_ui":
        print("SETTING TOOL SELECTOR SYSTEM PROMPT")
        system_prompt = toolselector_llm_ui
       
    response = client.chat.completions.create(
        model="gpt-4o", # model = "deployment_name".
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

def use_generic_llm(user_input, task="question", history=""):
    return call_azure_openai_gpt4(user_input, task, history)

def classifier(user_input, task="question", history=""):
    return call_azure_openai_gpt4(user_input, task, history)

def generalist(user_input, task="generate", history=""):
    return call_azure_openai_gpt4(user_input, task, history)

def toolselector(user_input, task, history=""):
    return call_azure_openai_gpt4(user_input, task, history)

def beamline_scientist(user_prompt, history, paper_directory):
    use_paperQA2_beamline(user_prompt, history, paper_directory)

def scientist(user_prompt, history, paper_directory):

    tool = toolselector(user_prompt, task = "toolselector", history = history)

    if tool == "Thorough":
        print("THOROUGH")
        response = use_paperQA2(user_prompt, history, paper_directory)

    if tool == "HighLevel":
        print("HIGHLEVEL")
        response = use_paperQA2_easy(user_prompt, history, paper_directory)

    if tool == "Beamline":
        print("BEAMLINE")
        response = use_paperQA2_beamline(user_prompt, history, paper_directory)

    return response

def scientist_ui(user_prompt, history, paper_directory):

    tool = toolselector(user_prompt, task = "toolselector_ui", history = history)

    if tool == "Thorough":
        print("THOROUGH")
        response = use_paperQA2(user_prompt, history, paper_directory)
        responder = "PaperQA"

    if tool == "HighLevel":
        print("HIGHLEVEL")
        response = use_paperQA2_easy(user_prompt, history, paper_directory)
        responder = "PaperQA_Lite"

    return response, responder

def use_paperQA2(user_prompt, history, paper_directory):

    answer = ask(
        user_prompt,
        settings=Settings(
            embedding="ollama/mxbai-embed-large",
            paper_directory=paper_directory,
            history=history,

        ),
    )

    # Important to convert this to a string otherwise run into some error which receiving data at HAL
    return str(answer.answer)

def use_paperQA2_easy(user_input, history, paper_directory):

    print("RUNNING EASY PAPERQA2 AGENT")

    settings = Settings(
            embedding="ollama/mxbai-embed-large",
            paper_directory = paper_directory,
            history = history,
            agent_type = "Fake"

    )

    settings.answer.evidence_k = 1
    settings.answer.answer_max_sources = 1
    settings.prompts.qa = papaerqa_lite

    answer = ask(
        user_input,
        settings=settings,
    )

    #Important to convert this to a string otherwise run into some error which receiving data at HAL
    return str(answer.answer)

def use_paperQA2_beamline(user_input, history, paper_directory):

    print("RUNNING EASY PAPERQA2 AGENT FOR BEAMLINE QUESTION")

    settings = Settings(
            embedding="ollama/mxbai-embed-large",
            paper_directory = paper_directory,
            history = history,
            agent_type = "Fake"

    )

    settings.answer.evidence_k = 1
    settings.answer.answer_max_sources = 1
    settings.prompts.qa = paperqa_lite_beamline

    settings.parsing.use_doc_details = False

    answer = ask(
        user_input,
        settings=settings,
    )

    #Important to convert this to a string otherwise run into some error which receiving data at HAL
    return str(answer.answer)

def make_semantic_router():
    encoder = HuggingFaceEncoder()

    operation = Route(
        name="Domain-Specific",
        utterances=[
            "What are the key research findings on graphene applications in energy storage?",
            "Can you provide a literature review on the toxicity of quantum dots in biological systems?",
            "What does the latest research say about nanoparticle synthesis methods for catalysis?",
            "Summarize the findings on nanomaterial interactions with biological tissues.",
            "What are the latest developments in nanotechnology for drug delivery systems?",
            "Which papers discuss the optical properties of nanomaterials for solar energy?",
            "What does the literature say about the environmental impact of nanomaterials?",
            "Can you provide a synthesis of research on the use of nanotechnology in treating cancer?",
            "What are the most cited studies on the mechanical properties of carbon nanotubes?",
            "Summarize the recent advances in nanoparticle-based sensors for medical diagnostics.",
            "Which nanoscience papers explore the effects of nanoparticles on soil ecosystems?",
            "What does recent research indicate about the scalability of nanomanufacturing techniques?",
            "Can you provide a summary of the research on using nanomaterials in next-generation batteries?",
            "What are the findings from recent studies on the toxicity of nanomaterials in aquatic environments?",
            "What research has been done on the thermal conductivity of nanomaterials for heat dissipation?",
            "Which studies explore the use of nanomaterials in flexible electronics?",
            "What do recent papers say about the role of nanotechnology in antimicrobial coatings?",
            "Can you review the latest studies on nanomaterial applications in tissue engineering?",
            "What findings are there on the stability of nanoparticles under different environmental conditions?",
            "Which studies have explored the use of quantum dots for single-photon emission applications?"
        ],
    )

    routes = [operation]

    rl = RouteLayer(encoder=encoder, routes=routes)

    return rl


def use_semantic_router(user_input):
    semantic_router = make_semantic_router()

    prompt_type = semantic_router(user_input).name

    if prompt_type is None:
        print("SEMANTIC ROUTER MISSED")
        prompt_type = use_generic_llm(user_input, task="question")

    return prompt_type

def use_naive_rag(user_input, history, docs_dir, persist_dir, force_reload = False):

    config = RAGConfig(
        azure_api_key=os.environ.get('AZURE_API_KEY'),  # Replace with your key
        azure_api_base=os.environ.get('AZURE_API_BASE'),  # Replace with your endpoint
        azure_deployment_name=os.environ.get('AZURE_DEPLOYMENT'),  # Replace with your deployment name
        docs_dir=docs_dir,  # Replace with your PDF directory
        persist_dir = persist_dir,
        # persist_dir="/home2/common/chroma_vectordb_dir/",
        force_reload = force_reload # Replace with where you want to save the vector DB
    )

    try:
        # Initialize RAG system
        rag = RAGSystem(config)
        print("RAG system initialized successfully!")
        answer = rag.query(user_input, history)
        return answer   

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    

# Add these imports at the top
from openai import AzureOpenAI
import os

# Add new functions for GPT-4o and o1
def use_gpt4o(prompt, history=""):

    client = AzureOpenAI(
            api_key = os.environ.get('AZURE_API_KEY'),  
            api_version='2023-05-15',
            azure_endpoint = os.environ.get('AZURE_API_BASE'),
            azure_deployment = os.environ.get('AZURE_DEPLOYMENT')
            )

    try:
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        
        # Add history if exists
        if history:
            # Parse history and add to messages
            for line in history.split('\n'):
                if line.startswith('Human: '):
                    messages.append({"role": "user", "content": line[7:]})
                elif line.startswith('AI: '):
                    messages.append({"role": "assistant", "content": line[4:]})
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model="gpt-4o",  # or your deployed model name for GPT-4
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error with GPT-4o: {str(e)}"
    

def use_o1(prompt, history=""):

    client = AzureOpenAI(
            api_key = os.environ.get('AZURE_o1_API_KEY'),  
            api_version='2024-12-01-preview',
            azure_endpoint = os.environ.get('AZURE_o1_API_BASE'),
            azure_deployment = os.environ.get('AZURE_o1_DEPLOYMENT')
            )
    try:
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        
        # Add history if exists
        if history:
            # Parse history and add to messages
            for line in history.split('\n'):
                if line.startswith('Human: '):
                    messages.append({"role": "user", "content": line[7:]})
                elif line.startswith('AI: '):
                    messages.append({"role": "assistant", "content": line[4:]})
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model="o1",  # or your deployed model name for o1
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error with o1: {str(e)}"