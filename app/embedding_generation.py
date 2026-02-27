import os
import logging
import warnings
import configparser
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Document processing (Docling)
from docling.datamodel.base_models import InputFormat
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

# LangChain components
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss

# --- Configuration & Logging ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EmbeddingGeneration")

class DocumentProcessor:
    """
    Handles multi-format document conversion, splitting, and vector store generation.
    Uses HuggingFace embeddings (nomic-ai/nomic-embed-text-v1) to match the
    embedding model used by the rest of the application (smart_iot_app.py, rag_service.py).
    """
    
    def __init__(self):
        # Initialize HuggingFace Embeddings (same config as smart_iot_app.py and embedding_service.py)
        model_name = "nomic-ai/nomic-embed-text-v1"
        model_kwargs = {
            'device': 'cpu',
            'trust_remote_code': True
        }
        encode_kwargs = {'normalize_embeddings': False}

        logger.info(f"Initializing HuggingFace embeddings with model: {model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        # Initialize Docling Converter for multiple formats
        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF, InputFormat.DOCX],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=StandardPdfPipeline,
                    backend=PyPdfiumDocumentBackend
                ),
                InputFormat.DOCX: WordFormatOption(),
            }
        )
        
        # Initialize Text Splitter (same as notebook: chunk_size=1000, chunk_overlap=100)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
        )

    def convert_file(self, file_path: Path) -> List[LCDocument]:
        """Converts a single file to a list of LangChain documents (chunks)."""
        try:
            logger.info(f"Converting: {file_path.name}")
            conv_result = self.converter.convert(file_path)
            
            if conv_result and conv_result.document:
                markdown_content = conv_result.document.export_to_markdown()
                doc = LCDocument(
                    page_content=markdown_content, 
                    metadata={"source": file_path.name, "path": str(file_path)}
                )
                chunks = self.text_splitter.split_documents([doc])
                logger.info(f"Successfully converted {file_path.name} into {len(chunks)} chunks.")
                return chunks
            else:
                logger.error(f"Failed to convert: {file_path.name}")
                return []
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {str(e)}")
            return []

    def create_vector_db_from_dir(self, dir_path: str, output_path: str):
        """Processes all documents in a directory and saves them to a FAISS index."""
        path = Path(dir_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")
            
        all_chunks = []
        # Support PDF and Word documents
        extensions = ['*.pdf', '*.docx']
        files_to_process = []
        for ext in extensions:
            files_to_process.extend(list(path.glob(ext)))
            
        if not files_to_process:
            logger.warning(f"No matching files found in {dir_path}")
            return

        logger.info(f"Found {len(files_to_process)} files to process.")
        
        for file_path in files_to_process:
            chunks = self.convert_file(file_path)
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.error("No content extracted from any files.")
            return

        logger.info(f"Total chunks across all documents: {len(all_chunks)}")

        # Initialize FAISS
        logger.info("Initializing FAISS index...")
        test_vector = self.embeddings.embed_query("warmup")
        dimension = len(test_vector)
        
        index = faiss.IndexFlatL2(dimension)
        vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

        logger.info("Generating embeddings and adding to vector store...")
        vector_store.add_documents(documents=all_chunks)
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving combined FAISS index to: {output_path}")
        vector_store.save_local(str(output_path))
        logger.info(f"Vector store saved successfully with {vector_store.index.ntotal} vectors.")
        return vector_store

def run_pipeline():
    """Main execution flow using project configuration."""
    load_dotenv()
    
    # Load project config
    config = configparser.ConfigParser()
    config_path = Path(__file__).resolve().parent / "utils" / ".config"
    
    if config_path.exists():
        config.read(str(config_path))
        vector_store_root = config.get("VECTOR_DB_PATH", "vector_db_path", fallback="./app/vectorstores")
        faiss_db_name = config.get("FAISS_DB_NAME", "faiss_db_name", fallback="smart_iot_vector_store")
    else:
        vector_store_root = "./app/vectorstores"
        faiss_db_name = "smart_iot_vector_store"

    project_root = Path(__file__).resolve().parent.parent
    
    # Source folder: all docx files from data_document
    data_dir = str(project_root / "data_document")
    
    # Target path for the vector store
    if vector_store_root.startswith("./"):
        full_vector_path = project_root / vector_store_root.lstrip("./") / faiss_db_name
    else:
        full_vector_path = Path(vector_store_root) / faiss_db_name
    
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Vector store output: {full_vector_path}")
    
    processor = DocumentProcessor()
    
    try:
        logger.info(f"Processing all documents in: {data_dir}")
        processor.create_vector_db_from_dir(data_dir, str(full_vector_path))
        logger.info("Pipeline completed successfully for all documents!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    run_pipeline()