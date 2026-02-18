import logging
import uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def get_logger():
    return logging.getLogger("RAG-APP")

def generate_request_id():
    return str(uuid.uuid4())[:8]
