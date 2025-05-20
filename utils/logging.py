import logging
import os
from datetime import datetime
from config.settings import LOGS_DIR

# Set up logging
os.makedirs(LOGS_DIR, exist_ok=True)
log_file = os.path.join(LOGS_DIR, f'app_{datetime.now().strftime("%Y%m%d")}.log')

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Create logger instance
logger = logging.getLogger('acadally_assistant')

def log_error(error, context=None):
    """Log an error with optional context."""
    if context:
        logger.error(f"Error in {context}: {error}")
    else:
        logger.error(f"Error: {error}")

def log_info(message):
    """Log an informational message."""
    logger.info(message)

def log_warning(message):
    """Log a warning message."""
    logger.warning(message)

def log_debug(message):
    """Log a debug message."""
    logger.debug(message)

def log_query(teacher_id, question, query=None, result=None):
    """Log a query with its results."""
    logger.info(f"Teacher ID: {teacher_id}")
    logger.info(f"Question: {question}")
    if query:
        logger.info(f"SQL Query: {query}")
    if result is not None:
        if hasattr(result, 'empty') and not result.empty:
            logger.info(f"Found {len(result)} results")
        else:
            logger.info("No results found")

def log_exception(e, context=""):
    """Log an exception with traceback."""
    import traceback
    logger.error(f"Exception in {context}: {e}")
    logger.error(traceback.format_exc())