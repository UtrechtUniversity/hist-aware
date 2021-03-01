import os
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseSettings
import src.utils.keywords as kw


class Settings(BaseSettings):
    # Data directory in base path
    DATA_DIR = "data"
    #  Data directory for saving files within DATA_DIR
    DATA_DIR_SAVE = "processed"
    # Data directory for raw files within DATA_DIR
    DATA_DIR_RAW = "raw"
    # Name of delpher directory
    DATA_DIR_DELPHER = "delpher"

    # Bool to ungizp metadata, do only once!
    UNGIZP = False
    # Decide whether to process and save articles and metadata data
    DATAFILE = {
        "start": "False",
        "metadata": "False",
        "articles": "False",
    }
    # Arguments to use for text search
    SEARCH_WORDS = False
    LIST_INCL_WORDS = kw.KEYWORDS_GAS
    LIST_EXCL_WORDS = kw.EXCL_WORDS_GAS
    PREPROCESS = False
    CLASSIFY = True

    # Selected topic and decade
    TOPIC = "olie"
    DECADE = "1970s"

    # If other decade than current used to classify with NB
    # set custom labels to True and select the decade below
    CUSTOM_LABELS = True

    class Config:
        case_sensitive = True


settings = Settings()
