import logging
import os

LOG_DIR = "./app/logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the desired logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "app.log")),  # Log to a file
        logging.StreamHandler()  # Also log to the console
    ],
)

# Logger instance
logger = logging.getLogger("RAGSystem")
