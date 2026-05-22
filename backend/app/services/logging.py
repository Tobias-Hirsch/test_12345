import os
import sys
import logging
from datetime import datetime
# Kommentar
from app.core.config import settings # Import settings

# Define a fixed log directory inside the container
log_dir = "/app/logs/api_log"
# The directory is already created by the Dockerfile, but we ensure it exists.
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"ai_log_{datetime.now().strftime('%Y%m%d')}.log")

# Kommentar
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
api_service_logger = logging.getLogger("api_service")
chat_base_logger = logging.getLogger("chat_base")
base_tools_logger = logging.getLogger("base_tools")
template_ai_create_logger = logging.getLogger("template_ai_create_logger")
pdf_process_content_logger = logging.getLogger("pdf_process_content_logger")
word_process_content_logger = logging.getLogger("word_process_content_logger")
xlsx_process_content_logger = logging.getLogger("xlsx_process_content_logger")