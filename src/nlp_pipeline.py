# nlp_pipeline.py
import os
from os.path import dirname
import logging

import nl_core_news_lg

# Import modules
from src import pipeline_text_selection
from src import logger

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)

# Setup logger
HAlogger = logger.get_logger("Pipeline")

FILE_PATH = dirname(dirname(os.path.realpath(__file__)))
# Data path for Delpher data
DIR_PATH = os.path.join(FILE_PATH, "data", "1950", "Delpher")
# Save path
SAVE_PATH = os.path.join(FILE_PATH, "data", "processed")
# Decide whether to ungizip metadata
UNGIZP = False
# Decide whether to process and save articles and metadata data
DATAFILE = False
# Keywords to use for the naive text selection
KEYWORDS = ["olie", "aardgas", "steenkool"]
# Number of synonyms to retrieve for each keyword, the more the less accurate
NUM_SYNONYMS = 20
# Transformer model to use for the creation of the synonyms
NLP = nl_core_news_lg.load()


if __name__ == "__main__":
    TextSelection = pipeline_text_selection.TextSelection(
        FILE_PATH=FILE_PATH,
        DIR_PATH=DIR_PATH,
        SAVE_PATH=SAVE_PATH,
        UNGIZP=UNGIZP,
        DATAFILE=DATAFILE,
        KEYWORDS=KEYWORDS,
        NUM_SYNONYMS=NUM_SYNONYMS,
        NLP=NLP,
    )
    # TODO: divide the extraction and save of data from the processing (e.g.)
    # searching synonyms
    # TODO: transform all these functions to manage `chunks` of data!

    # TextSelection.iterate_directories()
    # `Process_files` works only if DATAFILE is True
    # TextSelection.process_files()
    TextSelection.retrieved_saved_files()
    TextSelection.search_synonyms()
