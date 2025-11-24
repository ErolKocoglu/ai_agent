from pathlib import Path
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain.agents import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import INDEX_DIR, DOCS_DIR, EMBEDDING_MODEL_NAME, LLM_MODEL_NAME, LLM_TEMPERATURE

# Initialize Embedding Model
embedding_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
Settings.embed_model = embedding_model

# Initialize LLM
llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)
Settings.llm = llm

def build_or_load_index():
    if Path(INDEX_DIR).exists() and any(Path(INDEX_DIR).glob("*.json")):
        print("✅ Loading existing index...")
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        return load_index_from_storage(storage_context)

    print("⚡ No index found, creating a new one...")
    # Check if docs directory exists
    if not Path(DOCS_DIR).exists():
        print(f"⚠️ Docs directory not found at {DOCS_DIR}. Creating it.")
        Path(DOCS_DIR).mkdir(parents=True, exist_ok=True)
        return None # Or handle empty index gracefully

    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    
    if not documents:
        print("⚠️ No documents found in docs directory.")
        return None

    index = VectorStoreIndex.from_documents(
        documents, embed_model=embedding_model
    )
    index.storage_context.persist(persist_dir=INDEX_DIR)
    print("💾 Index created and saved.")
    return index

# Global index instance (lazy loading might be better but keeping simple for now)
index = build_or_load_index()
query_engine = index.as_query_engine() if index else None

def search_docs(query: str) -> str:
    """Gemini LLM + Gemini Embedding document search"""
    if not query_engine:
        return "Document search is unavailable because no index could be created."
        
    response = query_engine.query(query)
    answer = str(response)

    sources = []
    if hasattr(response, 'source_nodes'):
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
