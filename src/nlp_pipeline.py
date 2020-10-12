# nlp_pipeline.py

import logging

# Import modules
from src import pipeline_text_selection
from src import logger

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)

# Setup logger
HAlogger = logger.get_logger("Pipeline")

if __name__ == "__main__":
    pipeline_text_selection.main()
