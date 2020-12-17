# nlp_pipeline.py
import os
import sys
from os.path import dirname

from nlp_pipeline import PipelineArticles
from models.tfidf import ClassifyArticles
import utils.keywords as kw

from imblearn.over_sampling import (
    SMOTE,
)
from sklearn.naive_bayes import MultinomialNB

from loguru import logger

# Import modules
sys.path.insert(0, "..")


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
SEARCH_WORDS = False
KEYWORDS = kw.KEYWORDS_OLIE
EXCL_WORDS = kw.EXCL_WORDS_OLIE

PREPROCESS = False
CLASSIFY = True

TOPIC = "olie"
DECADE = "1980s"

if __name__ == "__main__":
    pipe = PipelineArticles(
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
        logger.info("Ungzip metadata")
        pipe.ungzip_metadata_files()

    # If folder file_info is empty, create list of files
    logger.info("Iterate directories")
    pipe.iterate_directories()

    logger.info("Process files")
    pipe.process_files()

    if SEARCH_WORDS is True:
        logger.info("Retrieved saved files")
        pipe.retrieved_saved_files()

        logger.info("Search synonyms")
        pipe.search_synonyms()

    if PREPROCESS is True:
        logger.info("Preprocess selected articles for labeling")
        pipe.process_selected_articles()

    if CLASSIFY is True:
        logger.info("Training classification model")
        ca = ClassifyArticles(SAVE_PATH=SAVE_PATH, DECADE=DECADE, TOPIC=TOPIC)
        pipe = ca.run_classification_pipeline(
            sampler=SMOTE(),
            classifier=MultinomialNB(),
        )
        ca.predict(pipe, THRESHOLD=0.15)
        ca.predict(pipe, THRESHOLD=0.10)
        ca.predict(pipe, THRESHOLD=0.05)
        ca.predict(pipe, THRESHOLD=0.02)
