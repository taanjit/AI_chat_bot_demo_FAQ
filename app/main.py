from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import shutil
import os
import logging
import os
import warnings
from app.routers import rag, uploads
from dotenv import load_dotenv
import logging

# Suppress logs from watchfiles
logging.getLogger("watchfiles.main").setLevel(logging.WARNING)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # if using chroma nd faiss in same environment
warnings.filterwarnings("ignore")

load_dotenv()


# Initialize FastAPI app
app = FastAPI(
    title="NetvirE SmartAssist 📚",
    version="1.0",
    description="Chat with NetvirE SmartAssist"
)

# Allow all origins with necessary methods and headers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(uploads.router, prefix="/uploads", tags=["Uploads"])
app.include_router(rag.router, prefix="/rag", tags=["RAG"])

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         "app.main:app",
#         host="0.0.0.0",
#         port=8088,
#         reload=True,
#         ws_ping_interval=20,   # Interval between pings in seconds
#         ws_ping_timeout=60,      # Timeout before considering connection inactive
#         timeout_keep_alive=60  # Timeout for keeping a connection alive
#     )   




