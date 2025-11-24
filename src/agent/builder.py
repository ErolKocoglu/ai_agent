from langchain.agents import initialize_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from llama_index.core import Settings

from src.config import LLM_MODEL_NAME, LLM_TEMPERATURE
from src.tools.document_search import doc_tool
from src.tools.joke_tool import joke_tool

def build_agent():
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE
    )
    Settings.llm = llm

    web_search_tool = DuckDuckGoSearchRun()

    agent = initialize_agent(
        tools=[doc_tool, joke_tool, web_search_tool],
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True,
        handle_parsing_errors=True
    )
    return agent
