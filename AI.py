"""
@reference: PE-GPT: a New Paradigm for Power Electronics Design, by Fanfan Lin, Xinze Li, et al.
@code-author: Xinze Li, Fanfan Lin
@github: https://github.com/XinzeLee/PE-GPT

@reference:
    Following references are related to power electronics GPT (PE-GPT)
    1: PE-GPT: a New Paradigm for Power Electronics Design
        Authors: Fanfan Lin, Xinze Li (corresponding), Weihao Lei, Juan J. Rodriguez-Andina, Josep M. Guerrero, Changyun Wen, Xin Zhang, and Hao Ma
        Paper DOI: 10.1109/TIE.2024.3454408
"""

import asyncio
from core.gui.gui import build_gui, init_states, display_history
from core.gui.design_stages import design_flow, task_agent
from core.llm.llm import ollama_init, rag_load

if __name__ == "__main__":
    
    FlexRes = True
    llm_model = "mistral"
    client = ollama_init(ollama_model="mistral", api_url="http://localhost:11434")
    build_gui()
        
    temperature, chunk_size, top_k = 0.1, 512, 7
    
    with open('core/knowledge/prompts/prompt.txt', 'r') as file:
        system_prompt = file.read()
    
    index0 = rag_load("core/knowledge/kb/database", llm_model, temperature=temperature, 
                       chunk_size=chunk_size)
    chat_engine0 = index0.as_chat_engine(chat_mode="context",
                                         similarity_top_k=top_k,
                                         system_prompt=system_prompt)
    
    index1 = rag_load("core/knowledge/kb/database1", llm_model, temperature=temperature, 
                       chunk_size=chunk_size)
    chat_engine1 = index1.as_chat_engine(similarity_top_k=top_k,
                                         system_prompt=system_prompt)
    
    index2 = rag_load("core/knowledge/kb/introduction", llm_model, temperature=temperature, 
                       chunk_size=chunk_size)
    chat_engine2 = index2.as_chat_engine(chat_mode="context",similarity_top_k=top_k)
    
    agent_intent = task_agent()
    
    initial_values = {key:None for key in ['M', 'Uin', 'Uo', 'P', 
                                           'fs', 'vp', 'vs', 'iL']}
    init_states(initial_values)
    
    display_history()

    agents = [chat_engine0, chat_engine1, chat_engine2, agent_intent]
    
    asyncio.run(design_flow(agents, client, FlexRes=FlexRes))
