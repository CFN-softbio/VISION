import streamlit as st
import sys
import os

# Add the project root to the sys paths, not preferred but necessary if we want to keep this directory structure
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.hal_beam_com.chatbot_utils import (

    scientist,
    classifier,
    generalist,
    scientist_ui

)

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

if "messages" not in st.session_state:
    st.session_state.messages = []

def about_page():
    st.subheader("About")   

def ask_llm_page():
    st.subheader("Rag App")

about = st.Page(about_page, title="About", icon=":material/info:")
ask_llm = st.Page(ask_llm_page, title="Ask LLM", icon=":material/chat:")

pg = st.navigation({"Home": [ask_llm], "About": [about]})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask LLM"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking ..."):
        prompt_type = classifier(prompt, history = format_chat_history(st.session_state.messages))

        match prompt_type:
            case "Generalist":
                print("GOING TO GENERALIST")
                bot_response = generalist(prompt, history = format_chat_history(st.session_state.messages))
                responder = "Generic GPT-4o"

            case "Scientist":
                print("GOING TO SCIENTIST")
                bot_response, responder = scientist_ui(prompt, history = format_chat_history(st.session_state.messages), paper_directory="/home2/smathur/RAG/CFN_publication_PDFs")    

    with st.chat_message("assistant"):
        # st.markdown(bot_response)
        st.markdown(f"{bot_response} \n\n Reponse generated from {responder}")

    st.session_state.messages.append({"role": "assistant", "content": f"{bot_response} \n\n Reponse generated from {responder}"})
