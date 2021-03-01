import os
import sys
from os.path import dirname

from imblearn.over_sampling import SMOTE
from loguru import logger
from sklearn.naive_bayes import MultinomialNB

from src.nlp_pipeline import PipelineArticles
from src.models.tfidf import ClassifyArticles
from src.core.config import settings
import src.utils.keywords as kw

# Import modules
sys.path.insert(0, "..")
# General directory path
FILE_PATH = dirname(dirname(os.path.realpath(__file__)))

# Use core.config to set this
DIR_PATH = os.path.join(
    FILE_PATH, settings.DATA_DIR, settings.DATA_DIR_RAW, settings.DATA_DIR_DELPHER
)
SAVE_PATH = os.path.join(FILE_PATH, settings.DATA_DIR, settings.DATA_DIR_SAVE)
UNGIZP = settings.UNGIZP
DATAFILE = settings.DATAFILE
SEARCH_WORDS = settings.SEARCH_WORDS
KEYWORDS = settings.LIST_INCL_WORDS
EXCL_WORDS = settings.LIST_EXCL_WORDS
PREPROCESS = settings.PREPROCESS
CLASSIFY = settings.CLASSIFY
TOPIC = settings.TOPIC
DECADE = settings.DECADE
CUSTOM_LABELS = settings.CUSTOM_LABELS

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
        CUSTOM_LABELS=CUSTOM_LABELS,
    )

    # Ungzip metadata files
    if UNGIZP is True:
        logger.info("Ungzip metadata")
        pipe.ungzip_metadata_files()

    if CLASSIFY is not True:
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
        ca = ClassifyArticles(
            SAVE_PATH=SAVE_PATH, DECADE=DECADE, TOPIC=TOPIC, CUSTOM_LABELS=CUSTOM_LABELS
        )
        pipe, thres = ca.run_classification_pipeline(
            sampler=SMOTE(),
            classifier=MultinomialNB(),
        )
        ca.predict(pipe, DECADE, THRESHOLD=0.90)
        ca.predict(pipe, DECADE, THRESHOLD=0.95)
        ca.predict(pipe, DECADE, THRESHOLD=0.98)
        ca.predict(pipe, DECADE, THRESHOLD=0.99)
