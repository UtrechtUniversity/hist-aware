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
        "start": "True",
        "metadata": "True",
        "files": "False",
    }
)
# Keywords to use for the naive text selection
KEYWORDS = [
    "aardolie",
    "petrolie",
    "petroleum",
    "aardgas",
    "steenkool",
    "bruinkool",
    "cokes",
    "kool",
]
# Number of synonyms to retrieve for each keyword, the more the less accurate
NUM_SYNONYMS = 10
# Transformer model to use for the creation of the synonyms
NLP = nl_core_news_lg.load()


if __name__ == "__main__":
    TextSelection = TextSelection(
        FILE_PATH=FILE_PATH,
        DIR_PATH=DIR_PATH,
        SAVE_PATH=SAVE_PATH,
        UNGIZP=UNGIZP,
        DATAFILE=DATAFILE,
        KEYWORDS=KEYWORDS,
        # NUM_SYNONYMS=NUM_SYNONYMS,
        NLP=NLP,
    )
    # TODO: stream the search for synonyms? it has good and bad sides.
    # TODO: transform all these functions to manage `chunks` of data!

    # Ungzip metadata files
    if UNGIZP is True:
        logger.debug("Ungzip metadata")
        TextSelection.ungzip_metadata_files()

    # If folder file_info is empty, create list of files
    logger.debug("Iterate directories")
    TextSelection.iterate_directories()

    logger.debug("Process files")
    TextSelection.process_files()

    # Retrieval and searching should go togther as it cannot be kept in memory
    # logger.debug("Retrieved saved files")
    # TextSelection.retrieved_saved_files()

    # logger.debug("Search synonyms")
    # TextSelection.search_synonyms()
