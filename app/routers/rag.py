# app/routers/retriever.py
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, field_validator
from app.services.rag_service import load_retriever, load_llm_model, build_rag_chain
from app.services.prompt_service import prompt_manager
from typing import List, Literal, Optional
from enum import Enum
from langchain.callbacks import tracing_v2_enabled
from langsmith import Client
from app.utils.logger import logger
import asyncio
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any, Dict
import uuid
import os
from langchain.chat_models import ChatOpenAI
from langchain_groq import ChatGroq

import asyncio
import uuid
from typing import Dict, List
from fastapi import WebSocket, WebSocketDisconnect
from langchain_core.messages import AIMessageChunk

from dotenv import load_dotenv
load_dotenv()

router = APIRouter()

client = Client()

class Framework(str, Enum):
    groq = "groq"
    openai = "openai"


class EmbeddingModel(str, Enum):
    ollama = "ollama"
    openai = "openai"

class RetrieverRequest(BaseModel):
    llm_framework:Framework = Framework.groq
    question:str

class ChatRequest(BaseModel):
    question:str
    llm_framework:Framework = Framework.groq
    # NOTE: "llama3-8b-8192" has been decommissioned on Groq.
    # Use a similar, currently supported 8B Llama 3.x model.
    model_name : str = "llama-3.1-8b-instant"


# Define available models for CPU and GPU
MODEL_SETS = {
    "groq": ["llama-3.1-8b-instant", "meta-llama/llama-4-scout-17b-16e-instruct"],
    "openai": ["gpt-3.5-turbo", "gpt-4"]  # Another large model for GPU
}



@router.post("/query_retriever")
def query_retriever(retriever_req: RetrieverRequest= Query(..., title="Retrieve the documents")):
    """
    Query the loaded retriever for a given question.
    """
    try:
        retriever = load_retriever(retriever_req.llm_framework)  # Load from cache or initialize if not already loaded
        with tracing_v2_enabled() as cb:
            doc_results = retriever.invoke(retriever_req.question) 
            url = cb.get_run_url()
        return {
            "status": "success",
            "tracing_url": url,
            "results": [{"content": doc.page_content, "metadata": doc.metadata} for doc in doc_results],
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    

# Custom callback handler to stream output to WebSocket
class WebSocketCallbackHandler1(BaseCallbackHandler):
    def __init__(self):
        self._websocket = None
        self._is_closed = True  # Start as closed
        self._lock = asyncio.Lock()
        self._stop_event = asyncio.Event()  # Event to signal cancellation

    async def set_websocket(self, websocket: WebSocket):
        """Update the WebSocket reference safely and reset closed state."""
        async with self._lock:
            self._stop_event.set()
            # Clear the old WebSocket reference
            if self._websocket is not None and self._websocket != websocket:
                logger.info(f"Clearing old WebSocket reference: {id(self._websocket)}")
                self._websocket = None

            # Set the new WebSocket reference
            self._websocket = websocket
            self._is_closed = False  # Reset closed state
            self._stop_event.clear()  # Reset the stop event
            logger.info(f"WebSocket updated to: {id(websocket)}")

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        async with self._lock:
            if self._is_closed or self._websocket is None or self._stop_event.is_set():
                logger.warning("WebSocket is closed, not set, or streaming is stopped. Token not sent.")
                return

            try:
                logger.info(f"Sending token: {token}")
                await self._websocket.send_text(token)
            except RuntimeError as e:  # Catch RuntimeError
                logger.error(f"RuntimeError while sending token: {e}")
                self._is_closed = True  # Mark as closed
            except Exception as e:
                logger.error(f"Error sending token: {e}")
                self._is_closed = True
    
    # async def on_llm_end(self, *args, **kwargs):
    #     async with self._lock:
    #         if self._is_closed or self._websocket is None:
    #             logger.warning("WebSocket is closed, not set, or streaming is stopped. Token not sent.")
    #             return

    #         try:
    #             await self._websocket.send_text(".DONE.")
    #         except RuntimeError as e:  # Catch RuntimeError
    #             logger.error(f"RuntimeError while sending token: {e}")
    #             self._is_closed = True  # Mark as closed
    #         except Exception as e:
    #             logger.error(f"Error sending token: {e}")
    #             self._is_closed = True


    async def stop_streaming(self):
        """Signal the callback handler to stop streaming tokens."""
        async with self._lock:
            self._stop_event.set()  # Set the stop event
            logger.info("Token streaming stopped.")

    async def close(self):
        """Mark WebSocket as closed and clear the WebSocket reference."""
        async with self._lock:
            self._is_closed = True
            self._websocket = None  # Clear the WebSocket reference
            self._stop_event.set()  # Ensure streaming is stopped
            logger.info("WebSocketCallbackHandler closed.")


# In-memory storage for chat history
chat_history1: Dict[str, List[dict]] = {}
active_connections1: Dict[str, WebSocketCallbackHandler1] = {}
active_tasks: Dict[str, asyncio.Task] = {}  # Store active generation tasks

def get_chat_history1(session_id: str) -> List[dict]:
    """Retrieve chat history for a session."""
    return chat_history1.get(session_id, [])

def add_to_chat_history1(session_id: str, message: dict):
    """Add a message to the chat history for a session."""
    if session_id not in chat_history1:
        chat_history1[session_id] = []
    chat_history1[session_id].append(message)


@router.websocket("/chat")
async def websocket_chat(websocket: WebSocket, sessionId: str = None):
    await websocket.accept()
    
    # Check if handler exists for this session
    if sessionId in active_connections1:
        logger.info(f"Reusing existing handler for session: {sessionId}")
        callback_handler = active_connections1[sessionId]

        if sessionId in active_tasks:
            logger.info(f"Stopping and deleting old LLM task for session: {sessionId}")
            await callback_handler.stop_streaming()  # Stop streaming immediately
            active_tasks[sessionId].cancel()  # Cancel running task
            try:
                await active_tasks[sessionId]  # Ensure it's stopped
            except asyncio.CancelledError:
                logger.info(f"Generation task for session {sessionId} was forcefully cancelled.")
            del active_tasks[sessionId]  # Remove it completely

    else:
        logger.info(f"Creating new handler for session: {sessionId}")
        callback_handler = WebSocketCallbackHandler1()
        active_connections1[sessionId] = callback_handler  # Store handler

    # Always update WebSocket reference
    await callback_handler.set_websocket(websocket)
    generation_task = None  # Store running LLM task
    history = get_chat_history1(sessionId)
    

    await websocket.send_json({"type": "history", "history": history})
    try:
        while True:
            data = await asyncio.wait_for(websocket.receive_json(), timeout=300)
            if data.get("type") == "new_chat":
                chat_history1[sessionId] = []
                await websocket.send_json({"type": "history", "history": []})
                continue

            # Process query
            generation_req = ChatRequest(**data)
            logger.info(f"Received query: {generation_req}")
            add_to_chat_history1(sessionId, {"role": "user", "content": generation_req.question})

            if generation_req.llm_framework == "groq":
                retriever = load_retriever(generation_req.llm_framework)
                groq_api_key = os.getenv("GROQ_API_KEY") # Or securely fetch it another way
                if not groq_api_key:
                    await websocket.send_json({"type": "error", "message": "GROQ API key not configured."})
                    continue
                try:
                    llm = ChatGroq(
                        groq_api_key=os.getenv("GROQ_API_KEY"),
                        model= generation_req.model_name,
                        streaming=True, 
                        callbacks=[callback_handler]
                    )

                except Exception as e:
                    logger.error(f"GROQ initialization failed: {str(e)}")
                    await websocket.send_json({"type": "error", "message": f"Error loading GROQ model: {e}"})
                    continue

            # Prepare the RAG chain and invoke it
            prompt_template = prompt_manager.get_prompt('default')
            rag_chain = build_rag_chain(retriever, llm, prompt_template)
            generation_task = asyncio.create_task(rag_chain.ainvoke(generation_req.question))


            active_tasks[sessionId] = generation_task  # Store the new task
            try:
                response = await generation_task  # Wait for response
                add_to_chat_history1(sessionId, {"role": "assistant", "content": response})
            except asyncio.CancelledError:
                logger.info(f"Generation task for session {sessionId} was cancelled.")
                continue

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {sessionId}")
        # Cancel LLM Generation if it's still running
        if sessionId in active_tasks:
            await callback_handler.stop_streaming()  # Stop streaming
            active_tasks[sessionId].cancel()
            try:
                await active_tasks[sessionId]  # Wait for the task to be canceled
            except asyncio.CancelledError:
                logger.info(f"Generation task for session {sessionId} was cancelled.")
            del active_tasks[sessionId]  # Remove the task from the active tasks
        await callback_handler.close()
    except RuntimeError as e:  # Catch RuntimeError in WebSocket handling
        logger.error(f"RuntimeError in WebSocket connection: {e}")
    except Exception as e:
        logger.error(f"Unexpected WebSocket error: {e}")
        if sessionId in active_tasks:
            await callback_handler.stop_streaming()  # Stop streaming
            active_tasks[sessionId].cancel()
            try:
                await active_tasks[sessionId]  # Wait for the task to be canceled
            except asyncio.CancelledError:
                logger.info(f"Generation task for session {sessionId} was cancelled.")
            del active_tasks[sessionId]  # Remove the task from the active tasks

        await callback_handler.close()
        await websocket.close(code=1011)