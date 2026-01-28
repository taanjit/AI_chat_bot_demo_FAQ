
from fastapi import APIRouter, UploadFile, HTTPException, Query
import os
import shutil
from typing import List
from app.utils.logger import logger
from app.services import document_service, embedding_service
from enum import Enum
from pydantic import BaseModel
from typing import List, Literal

router = APIRouter()

UPLOAD_DIRECTORY = "./app/uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
ALLOWED_FILE_TYPES = ["application/pdf"]


class PDFProcessingTool(str, Enum):
    pymupdf = "pymupdf"

class Framework(str, Enum):
    groq = "groq"
    openai = "openai"

class PDFProcessingRequest(BaseModel):
    embedding_model: Framework = Framework.groq
    files: List[str]
    DB_name: str = "Thinkpalm"

ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".xlsx", ".pptx",
    ".md", ".adoc",
    ".html", ".xhtml",
    ".csv",
    ".png", ".jpeg", ".jpg", ".tiff", ".bmp", ".webp"
}

# Helper function to save files
def save_file(file: UploadFile):
    file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return file_path


@router.post("/upload")
async def upload_files(files: List[UploadFile]):
    file_names = []
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            logger.warning(f"Attempted to upload unsupported file type: {file.filename}")
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.filename}. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        filename = save_file(file)
        file_names.append(filename)
        logger.info(f"File '{file.filename}' uploaded successfully.")

    return {"message": "Success", "files": file_names}


@router.post("/process_files/")
async def process_uploaded_pdfs(pdf_request: PDFProcessingRequest):
    """
    Extract PDF to text and store as embeddings.
    """
    try:
        logger.info(f"Received request to process PDFs: {pdf_request.model_dump()}")
        if not pdf_request.files or len(pdf_request.files) == 0:
            logger.warning("No files provided for processing.")
            raise HTTPException(status_code=400, detail="No files provided for processing.")

        pdfs = []
        for file in pdf_request.files:
            file_path = os.path.join(UPLOAD_DIRECTORY, file)
            pdfs.append(file_path)
        
        pdf_chunks = document_service.process_pdf_with_tool(pdf_lst=pdfs) 
        embedding_service.create_embeddings(pdf_chunks, pdf_request.embedding_model, pdf_request.DB_name)
        logger.info("PDFs processed and embeddings generated successfully.")
        return {
            "status": "success",
            "emb db name": pdf_request.DB_name,
        }
    except Exception as e:
        logger.error(f"Error in processing PDFs: {e}")
        raise HTTPException(status_code=500, detail=f"Error in processing PDFs: {str(e)}")

        
@router.get("/get_uploaded_files/")
async def get_uploaded_files():
    files =[]
    for file in os.listdir(UPLOAD_DIRECTORY):
        files.append(file)
    return  {"message": "Success", "files": files}


@router.delete("/delete-file/{file_name}")
async def delete_file(file_name: str):
    file_path = os.path.join(UPLOAD_DIRECTORY, file_name)
        
    # Check if the file exists
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"File '{file_name}' not found.")

    try:
        os.remove(file_path)  # Delete the file
        return {"message": f"File '{file_name}' has been deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while deleting the file: {e}")