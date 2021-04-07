import os
import sys
from os.path import dirname
from pathlib import Path

from imblearn.over_sampling import SMOTE
from loguru import logger
from sklearn.naive_bayes import MultinomialNB

from src.nlp_pipeline import PipelineArticles
from src.models.tfidf import ClassifyArticles
from src.core.config import Settings
import src.utils.keywords as kw
from src.utils import utils

# Import modules
sys.path.insert(0, "..")

if __name__ == "__main__":
    settings = Settings()

    # General directory path
    FILE_PATH = Path(dirname(dirname(os.path.realpath(__file__))))

    # Use core.config to set this
    DIR_PATH = Path(
        os.path.join(
            FILE_PATH,
            settings.DATA_DIR,
            settings.DATA_DIR_RAW,
            settings.DATA_DIR_DELPHER,
        )
    )
    SAVE_PATH = Path(os.path.join(FILE_PATH, settings.DATA_DIR, settings.DATA_DIR_SAVE))

    pipe = PipelineArticles(
        FILE_PATH=FILE_PATH,
        DIR_PATH=DIR_PATH,
        SAVE_PATH=SAVE_PATH,
        UNGIZP=settings.UNGIZP,
        DATAFILE=settings.DATAFILE,
        KEYWORDS=settings.LIST_INCL_WORDS,
        EXCL_WORDS=settings.LIST_EXCL_WORDS,
        TOPIC=settings.TOPIC,
        DECADE=settings.DECADE,
        CUSTOM_LABELS=settings.CUSTOM_LABELS,
    )

    # Ungzip metadata files
    if settings.UNGIZP is True:
        logger.info("Ungzip metadata")
        pipe.ungzip_metadata_files()

    if settings.CLASSIFY is False:
        # If folder file_info is empty, create list of files
        logger.info("Iterate directories")
        pipe.iterate_directories()

        logger.info("Process files")
        pipe.process_files()

    if settings.MERGE is True:
        logger.info("Retrieved saved files")
        pipe.retrieved_saved_files()

        logger.info("Merge articles and metadata")
        pipe.merge_metadata_articles()

    if settings.SEARCH_WORDS is True:
        logger.info("Select articles using keywords")
        pipe.search_synonyms()

    if settings.PREPROCESS is True:
        logger.info("Preprocess selected articles for labeling")
        pipe.process_selected_articles()

    if settings.CLASSIFY is True:
        logger.info("Training classification model")
        ca = ClassifyArticles(
            SAVE_PATH=SAVE_PATH,
            DECADE=settings.DECADE,
            TOPIC=settings.TOPIC,
            CUSTOM_LABELS=settings.CUSTOM_LABELS,
            CUSTOM_DECADES=settings.CUSTOM_DECADES,
        )
        pipe, thres = ca.run_classification_pipeline(
            sampler=SMOTE(),
            classifier=MultinomialNB(),
        )
        ca.predict(pipe, settings.DECADE, THRESHOLD=0.90)
        ca.predict(pipe, settings.DECADE, THRESHOLD=0.95)
        ca.predict(pipe, settings.DECADE, THRESHOLD=0.98)
        ca.predict(pipe, settings.DECADE, THRESHOLD=0.99)

        utils.make_noise()
