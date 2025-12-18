import os
from pathlib import Path
import requests
from src.tools.math_tool import get_math_tool
from src.tools.time_tool import time_tool
from src.tools.wikipedia_tool import get_wikipedia_tool
from src.tools.python_repl_tool import get_python_repl_tool
from src.tools.file_tool import get_file_tools

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings
)
#from llama_index.embeddings.gemini import GeminiEmbedding
from langchain.agents import Tool, initialize_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_community.tools import DuckDuckGoSearchRun


load_dotenv()

INDEX_DIR = "./index"
DOCS_DIR = "./docs"

embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.embed_model = embedding_model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

Settings.llm = llm

def build_or_load_index():
    if Path(INDEX_DIR).exists() and any(Path(INDEX_DIR).glob("*.json")):
        print("✅ Loading existing index...")
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        return load_index_from_storage(storage_context)

    print("⚡ No index found, creating a new one...")

    documents = SimpleDirectoryReader(DOCS_DIR).load_data()

    

    index = VectorStoreIndex.from_documents(
        documents, embed_model=embedding_model
    )
    index.storage_context.persist(persist_dir=INDEX_DIR)
    print("💾 Index created and saved.")
    return index


index = build_or_load_index()
query_engine = index.as_query_engine()


# ------------------------------
# 3️⃣ Tool-1: Document Search (Query Engine)
# ------------------------------
def search_docs(query: str) -> str:
    """Gemini LLM + Gemini Embedding document search"""
    response = query_engine.query(query)
    answer = str(response)

    
    sources = []
    for node in response.source_nodes:
        fname = node.node.metadata.get("file_name", "unknown")
        sources.append(fname)

    if sources:
        answer += f"\n\n🔗 Sources: {', '.join(set(sources))}"
    return answer


doc_tool = Tool(
    name="docs_search",
    func=search_docs,
    description="Use this tool to answer questions about large language models.",
)


# ------------------------------
# 4️⃣ Tool-2: API Call 
# ------------------------------
def get_joke(_: str) -> str:
    """Returns a random joke from an API"""
    try:
        response = requests.get("https://icanhazdadjoke.com/", headers={"Accept": "application/json"})
        return response.json().get("joke", "No joke found.")
    except Exception as e:
        return f"API error: {e}"


joke_tool = Tool(
    name="get_joke",
    func=get_joke,
    description="Use this tool to get a random joke.",
)

web_search_tool = DuckDuckGoSearchRun()

# ------------------------------
# 5️⃣ Agent (Gemini LLM)
# ------------------------------




# ... (existing imports)

# Initialize new tools
math_tool = get_math_tool(llm)
wikipedia_tool = get_wikipedia_tool()
python_repl_tool = get_python_repl_tool()
file_tools = get_file_tools()

all_tools = [doc_tool, joke_tool, web_search_tool, math_tool, time_tool, wikipedia_tool, python_repl_tool] + file_tools

agent = initialize_agent(
    tools=all_tools,
    llm=llm,
    agent="structured-chat-zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True
)


# ------------------------------
# 6️⃣ Kullanıcı Sorgusu
# ------------------------------
if __name__ == "__main__":
    while True:
        query = input("\n💬 User Query: ")
        if query.lower() in {"exit", "quit"}:
            print("👋 Exiting...")
            break
        response = agent.invoke(query)
        print("\n--- Final Answer ---")
        print(response)
