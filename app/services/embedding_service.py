from langchain_ollama import OllamaEmbeddings
import os
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from app.utils.logger import logger
from langchain_huggingface import HuggingFaceEmbeddings


VECTOR_STORE_PATH = "./app/vectorstores/faiss_index"
# def load_ollama_embeddings():
#     embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")
#     single_vector = embeddings.embed_query("this is some text data")
#     index = faiss.IndexFlatL2(len(single_vector))
#     logger.info(f"Initialized Ollama embedding model {embeddings.model}.")
#     return embeddings, index

def load_huggingface_embeddings():
    model_name = "nomic-ai/nomic-embed-text-v1"  # Updated to the actual model from error
    model_kwargs = {
        'device': 'cpu',
        'trust_remote_code': True  # Allow custom code execution
    }
    encode_kwargs = {'normalize_embeddings': False}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    # Test embedding to get vector dimension
    single_vector = embeddings.embed_query("this is some text data")
    index = faiss.IndexFlatL2(len(single_vector))
    
    logger.info(f"Initialized Hugging Face embedding model {model_name}.")
    return embeddings, index

def create_embeddings(pdf_chunks, embedding_model, db_name):
    embeddings, index = None, None
    if embedding_model == "groq":
        embeddings, index = load_huggingface_embeddings()
    
    if embeddings and index:
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        # store documents
        ids = vector_store.add_documents(documents=pdf_chunks)
        vector_store.index_to_docstore_id
        vector_store.save_local(os.path.join(VECTOR_STORE_PATH,db_name))
        logger.info("Vector store saved successfully.")

