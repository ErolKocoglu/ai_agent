from langchain.agents import Tool
from langchain.chains import LLMMathChain
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import LLM_MODEL_NAME, LLM_TEMPERATURE

def get_math_tool(llm=None):
    if llm is None:
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)
    
    math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    
    return Tool(
        name="Calculator",
        func=math_chain.run,
        description="Useful for when you need to answer questions about math."
    )
