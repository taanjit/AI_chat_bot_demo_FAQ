# app/services/retriever_service.py
from langchain.vectorstores import FAISS
import os
# from app.services.embedding_service import load_ollama_embeddings
from app.services.embedding_service import load_huggingface_embeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser # GETTING FINAL OUT AS STRING
from langchain_core.runnables import RunnablePassthrough #parse question and context directly to LLM
from langchain_core.prompts import ChatPromptTemplate #to pass prompt with context (chunk of data)
from langchain_core.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.base import AsyncCallbackHandler
from langchain_ollama import ChatOllama
from app.utils.logger import logger
import configparser


config = configparser.ConfigParser()
# Load the config file
config.read("./app/utils/.config")

# Access values
loaded_models = {}  # Global cache for LLM models

VECTOR_STORE_PATH = config["VECTOR_DB_PATH"]["vector_db_path"]
faiss_db_name = config["FAISS_DB_NAME"]["faiss_db_name"]

retrievers = {}  # Global dictionary to cache retrievers
llms = {}

#combine all the chunk to single text doc to pass it as single context
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def load_retriever(llm_framework:str):
    """
    Load a retriever from the FAISS index for the given database name.
    If the retriever is already loaded, return the cached instance.
    """
    if faiss_db_name in retrievers:
        return retrievers[faiss_db_name]  # Return cached retriever

    embedding_path = os.path.join(VECTOR_STORE_PATH, faiss_db_name)
    logger.info(f"Loading retriever for database '{faiss_db_name}' from path '{embedding_path}'")

    if not os.path.exists(embedding_path):
        raise ValueError(f"FAISS index not found for database '{faiss_db_name}' at path '{embedding_path}'")

    # Load embedding model
    if llm_framework == "groq":
        # embeddings, index = load_ollama_embeddings()
        embeddings, index = load_huggingface_embeddings()

    # Load FAISS retriever
    new_vector_store =  FAISS.load_local(embedding_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    
    retriever = new_vector_store.as_retriever(search_type="mmr",
                                              search_kwargs = {'k': 3, 
                                                                'fetch_k': 100,
                                                                'lambda_mult': 1})
    # Cache the retriever
    retrievers[faiss_db_name] = retriever

    return retriever

# app/services/model_service.py


def load_llm_model(model_name: str, callback_manager, base_url: str = "http://localhost:11434") -> ChatOllama:
    """
    Load an LLM model. If the model is already loaded, return the cached instance.
    """
    if model_name in loaded_models:
        return loaded_models[model_name]  # Return cached model
    
    # Initialize the LLM with streaming enabled
    model = ChatOllama(
        model=model_name,
        base_url=base_url,
        streaming=True,  # Enable streaming
        callback_manager=callback_manager,
    )
    logger.info("Initialized LLM model.")
    # Cache the model
    loaded_models[model_name] = model
    return model

def build_rag_chain(retriever_model, llm_model, prompt):
    prompt = ChatPromptTemplate.from_template(prompt)
    rag_chain = (
        {"context": retriever_model|format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm_model
        | StrOutputParser()
    )
    logger.info("Building RAG chain.")
    return rag_chain
