# -*- coding: utf-8 -*-
"""
@reference: PE-GPT: a New Paradigm for Power Electronics Design, by Fanfan Lin, Xinze Li, et al.
@code-author: Xinze Li, Fanfan Lin, and Weihao Lei
@github: https://github.com/XinzeLee/PE-GPT

@reference:
    Following references are related to power electronics GPT (PE-GPT)
    1: PE-GPT: a New Paradigm for Power Electronics Design
        Authors: Fanfan Lin, Xinze Li (corresponding), Weihao Lei, Juan J. Rodriguez-Andina, Josep M. Guerrero, Changyun Wen, Xin Zhang, and Hao Ma
        Paper DOI: 10.1109/TIE.2024.3454408
"""

import streamlit as st
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex,  SimpleDirectoryReader
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.node_parser import SimpleNodeParser

#Code ollama_init moi sinh ra client thoi, khong "nho" noi dung hay thuc hien cac tac vu phuc tap nao

def ollama_init(ollama_model="mistral", api_url="http://localhost:11434"):
    """
        initialize Ollama
    """    
    
    # Local model (Mistral) run on local link http://localhost:11434
    
    
    client = Ollama(model=ollama_model, 
                    base_url=api_url,
                    request_timeout=120)
    
    # model selection
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = ollama_model
    
    return client


# code nay chuyen du lieu o folder database thanh index, va cache lai de lan sau load nhanh hon, khong can lam lai tu dau nua
@st.cache_resource(show_spinner=False)
def rag_load(database_folder, llm_model, 
              temperature=None, chunk_size=None, system_prompt=None):
    """
        This function is the retrieval-augmented generation (RAG) for LLM
    """
    
    if chunk_size is None: chunk_size = 512
    if temperature is None: temperature = 0.0
    
    with st.spinner(text="Loading and indexing  docs – hang tight! This should take 1-2 minutes."):
        
        llm= Ollama(model=llm_model, 
                    base_url="http://localhost:11434", 
                    request_timeout=120, 
                    temperature=temperature,
                    system_prompt=system_prompt)
        # define the embedding model
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # define the service context
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size = chunk_size

        # load documents and create an index
        documents = SimpleDirectoryReader(database_folder).load_data()
        index = VectorStoreIndex.from_documents(documents)
        # set the system prompt for the index
        # if system_prompt is not None:
        #     index.set_system_prompt(system_prompt)
    return index


def get_msg_history():
    """
        get the message history 
    """
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        
    messages_history = [ChatMessage(role=MessageRole.USER 
                                    if msg["role"] == "user" 
                                    else MessageRole.ASSISTANT, 
                                    content=msg["content"])
                        for msg in st.session_state.messages]
    return messages_history




