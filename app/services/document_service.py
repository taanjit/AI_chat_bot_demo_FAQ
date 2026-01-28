import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.utils.logger import logger
from docling.datamodel.base_models import InputFormat
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from langchain_core.documents import Document as LCDocument


def convert_files_docling(input_paths):
    doc_converter = DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.IMAGE,
            InputFormat.DOCX,
            InputFormat.HTML,
            InputFormat.PPTX,
            InputFormat.ASCIIDOC,
            InputFormat.MD,
        ],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline, backend=PyPdfiumDocumentBackend
            ),
            InputFormat.DOCX: WordFormatOption(
                pipeline_cls=SimplePipeline
            ),
        },
    )

    if isinstance(input_paths, str):
        input_paths = [input_paths]

    results = doc_converter.convert_all(input_paths)

    if not results:
        logger.warning("[Warning] No documents converted. Check file format and path.")
        return []

    lc_docs = []
    for result in results:
        try:
            if result.document is not None:
                content = result.document.export_to_markdown()
                lc_docs.append(LCDocument(page_content=content))
            else:
                logger.warning(f"[Warning] Document conversion returned None for {result}")
        except Exception as e:
            logger.error(f"[Error] Failed to process result: {e}")

    return lc_docs



def process_pdf_with_tool(pdf_lst):

    lc_documents = convert_files_docling(pdf_lst)

    if lc_documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(lc_documents)
        return chunks
