from langchain.agents import initialize_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from llama_index.core import Settings
from langchain.memory import ConversationBufferMemory

from src.config import LLM_MODEL_NAME, LLM_TEMPERATURE
from src.tools.document_search import doc_tool
from src.tools.joke_tool import joke_tool

# Initialize new tools
    from src.tools.math_tool import get_math_tool
    from src.tools.time_tool import time_tool
    from src.tools.wikipedia_tool import get_wikipedia_tool
    from src.tools.python_repl_tool import get_python_repl_tool
    from src.tools.file_tool import get_file_tools
    from src.tools.pdf_tool import get_pdf_tool
    from src.tools.data_tool import get_data_tools
    # from src.tools.youtube_tool import youtube_tool
    # from src.tools.arxiv_tool import get_arxiv_tool
    from src.agent.prompts import SYSTEM_PROMPT

def build_agent():
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE
    )
    Settings.llm = llm

    web_search_tool = DuckDuckGoSearchRun()
    

    math_tool = get_math_tool(llm)
    wikipedia_tool = get_wikipedia_tool()
    python_repl_tool = get_python_repl_tool()
    file_tools = get_file_tools()
    pdf_tool = get_pdf_tool()
    data_tools = get_data_tools()
    # arxiv_tool = get_arxiv_tool()

    all_tools = [
        doc_tool, 
        joke_tool, 
        web_search_tool, 
        math_tool, 
        time_tool, 
        wikipedia_tool, 
        python_repl_tool,
        pdf_tool,
        # youtube_tool,
        # arxiv_tool
    ] + file_tools + data_tools

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = initialize_agent(
        tools=all_tools,
        llm=llm,
        agent="structured-chat-zero-shot-react-description",
        verbose=True,
        handle_parsing_errors=True,
        memory=memory,
        agent_kwargs={
            "prefix": SYSTEM_PROMPT,
            "memory_prompts": [SYSTEM_PROMPT],
            "input_variables": ["chat_history", "input", "agent_scratchpad"]
        }
    )
    return agent
