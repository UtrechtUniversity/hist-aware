# nlp_pipeline.py
from itertools import filterfalse
import os
import sys
from os.path import dirname

from loguru import logger
import nl_core_news_lg

# Import modules
sys.path.insert(0, "..")

from text_search import TextSearch
import keywords as kw

# General directory path
FILE_PATH = dirname(dirname(os.path.realpath(__file__)))
# Data path for Delpher data
DIR_PATH = os.path.join(FILE_PATH, "data", "raw", "delpher")
# Save path
SAVE_PATH = os.path.join(FILE_PATH, "data", "processed")
# Decide whether to ungizip metadata
UNGIZP = False
# Decide whether to process and save articles and metadata data
DATAFILE = dict(
    {
        "start": "False",
        "metadata": "False",
        "files": "False",
    }
)
# Arguments to use for text search
SEARCH_WORDS = True
KEYWORDS = kw.KEYWORDS_GAS
EXCL_WORDS = kw.EXCL_WORDS_GAS

TOPIC = "gas"
DECADE = "1970s"

if __name__ == "__main__":
    ts = TextSearch(
        FILE_PATH=FILE_PATH,
        DIR_PATH=DIR_PATH,
        SAVE_PATH=SAVE_PATH,
        UNGIZP=UNGIZP,
        DATAFILE=DATAFILE,
        KEYWORDS=KEYWORDS,
        EXCL_WORDS=EXCL_WORDS,
        TOPIC=TOPIC,
        DECADE=DECADE,
    )

    # Ungzip metadata files
    if UNGIZP is True:
        logger.debug("Ungzip metadata")
        ts.ungzip_metadata_files()

    # If folder file_info is empty, create list of files
    logger.debug("Iterate directories")
    ts.iterate_directories()

    logger.debug("Process files")
    ts.process_files()

    if SEARCH_WORDS is True:
        logger.debug("Retrieved saved files")
        ts.retrieved_saved_files()

        logger.debug("Search synonyms")
        ts.search_synonyms()

    logger.debug("Preprocess selected articles for labeling")
    ts.process_selected_articles()
