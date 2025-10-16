import streamlit as st
import sys
import os
import re
from enum import Enum

# Add the project root to the sys paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.hal_beam_com.chatbot_utils import (
    classifier,
    generalist,
    scientist_ui,
    use_naive_rag,
    use_paperQA2,
    use_gpt4o,
    use_o1
)

# Hide the sidebar by default
st.set_page_config(initial_sidebar_state="collapsed")

class ClassifierOutput(Enum):
    GENERALIST = "Generalist"
    SCIENTIST = "Scientist"
    SCHOLAR = "Scholar"

def format_chat_history(chat_history):
    has_assistant = any(message["role"] == "assistant" for message in chat_history)
    
    if not has_assistant:
        return ""
    
    formatted_history = ""
    for message in chat_history:
        if message["role"] == "user":
            formatted_history += f"Human: {message['content']}\n"
        elif message["role"] == "assistant":
            formatted_history += f"AI: {message['content']}\n"
    return formatted_history

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize RAG mode in session state if not exists
if "rag_mode" not in st.session_state:
    st.session_state.rag_mode = "Naive RAG + PaperQA"

def about_page():
    st.subheader("About")   

def ask_llm_page():
    st.subheader("Rag App")

def process_prompt_type(classifer_response):
    answer_match = re.search(r'<answer>(.*?)</answer>', classifier_response)
    if answer_match:
        prompt_type = answer_match.group(1)
    else:
        # Fallback in case tags aren't found
        prompt_type = "Generalist"
    return prompt_type

# about = st.Page(about_page, title="About", icon=":material/info:")
# ask_llm = st.Page(ask_llm_page, title="Ask LLM", icon=":material/chat:")

# pg = st.navigation({"Home": [ask_llm], "About": [about]})

# Add sidebar for RAG mode selection
with st.sidebar:
    st.session_state.rag_mode = st.radio(
        "Select Mode:",
        ["Naive RAG + PaperQA", "Only Naive RAG", "GPT-4o", "o1"]
    )

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask LLM"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking ..."):

        force_reload = False
        PAPER_DIR = "/home2/common/Chatbot_Papers"
        PERSIST_DIR_NAIVE = "/home2/common/chroma_vectordb_dir/"

        if st.session_state.rag_mode == "GPT-4o":
            bot_response = use_gpt4o(prompt, history=format_chat_history(st.session_state.messages))
            responder = "GPT-4o"
        
        elif st.session_state.rag_mode == "o1":
            bot_response = use_o1(prompt, history=format_chat_history(st.session_state.messages))
            responder = "o1"

        elif st.session_state.rag_mode == "Only Naive RAG":
            # Original workflow
            classifier_response = classifier(prompt, history=format_chat_history(st.session_state.messages))
            prompt_type = process_prompt_type(classifier_response)
            print(prompt_type)

            match prompt_type:
                case ClassifierOutput.GENERALIST.value:
                    print("GOING TO GENERALIST")
                    bot_response = generalist(prompt, history=format_chat_history(st.session_state.messages))
                    responder = "Generic GPT-4o"

                case ClassifierOutput.SCIENTIST.value:
                    print("GOING TO SCIENTIST")
                    bot_response = use_naive_rag(prompt, history=format_chat_history(st.session_state.messages), docs_dir = PAPER_DIR, persist_dir = PERSIST_DIR_NAIVE, force_reload = force_reload)
                    responder = 'Naive RAG'
                    
                case ClassifierOutput.SCHOLAR.value:
                    print("GOING TO SCIENTIST")
                    bot_response = use_naive_rag(prompt, history=format_chat_history(st.session_state.messages), docs_dir = PAPER_DIR, persist_dir = PERSIST_DIR_NAIVE, force_reload = force_reload)
                    responder = 'Naive RAG'

        else:
            classifier_response = classifier(prompt, history=format_chat_history(st.session_state.messages))
            prompt_type = process_prompt_type(classifier_response)

            match prompt_type:
                case ClassifierOutput.GENERALIST.value:
                    print("GOING TO GENERALIST")
                    bot_response = generalist(prompt, history=format_chat_history(st.session_state.messages))
                    responder = "Generic GPT-4o"

                case ClassifierOutput.SCIENTIST.value:
                    print("GOING TO SCIENTIST")
                    # New workflow for Naive RAG + PaperQA
                    # Try Naive RAG first
                    naive_response = use_naive_rag(prompt, history=format_chat_history(st.session_state.messages), docs_dir = PAPER_DIR, persist_dir = PERSIST_DIR_NAIVE, force_reload = force_reload)
                    
                    # If Naive RAG doesn't have enough information, use scientist_ui
                    if "I don't have enough information to answer this question." in naive_response:                        
                        print("NAIVE RAG INSUFFICIENT, GOING TO PAPER QA")
                        bot_response = use_paperQA2(prompt, history = format_chat_history(st.session_state.messages), paper_directory=PAPER_DIR)
                        responder = 'PaperQA'
                    else:
                        bot_response = naive_response
                        responder = 'Naive RAG'

                case ClassifierOutput.SCHOLAR.value:
                    bot_response = use_paperQA2(prompt, history = format_chat_history(st.session_state.messages), paper_directory=PAPER_DIR)
                    responder = 'PaperQA'
                                
    with st.chat_message("assistant"):
        st.markdown(f"{bot_response} \n\n Response generated from {responder}")

    st.session_state.messages.append({"role": "assistant", "content": f"{bot_response} \n\n Response generated from {responder}"})
