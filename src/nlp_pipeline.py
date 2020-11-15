# nlp_pipeline.py
from itertools import filterfalse
import os
import sys
from os.path import dirname

from loguru import logger
import nl_core_news_lg

# Import modules
sys.path.insert(0, "..")

from pipeline_text_selection import TextSelection

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
# Keywords to use for the naive text selection
KEYWORDS = [
    "aardolie",
    "petrolie",
    "petroleum",
    "olie",
    # "aardgas",
    # "gas",
    # "steenkool",
    # "bruinkool",
    # "cokes",
    # "kool",
]
EXCL_WORDS = [
    "wortel",
    "groenten",
    "vrucht",
]
TOPIC = "oil"

DECADE = "1990s"

if __name__ == "__main__":
    TextSelection = TextSelection(
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
    # TODO: transform all these functions to manage `chunks` of data?

    # Ungzip metadata files
    if UNGIZP is True:
        logger.debug("Ungzip metadata")
        TextSelection.ungzip_metadata_files()

    # If folder file_info is empty, create list of files
    logger.debug("Iterate directories")
    TextSelection.iterate_directories()

    logger.debug("Process files")
    TextSelection.process_files()

    logger.debug("Retrieved saved files")
    TextSelection.retrieved_saved_files()

    logger.debug("Search synonyms")
    TextSelection.search_synonyms()
