import os
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseSettings
import src.utils.keywords as kw


class Settings(BaseSettings):
    # Data directory in base path
    DATA_DIR: str = "data"
    #  Data directory for saving files within DATA_DIR
    DATA_DIR_SAVE: str = "processed"
    # Data directory for raw files within DATA_DIR
    DATA_DIR_RAW: str = "raw"
    # Name of delpher directory
    DATA_DIR_DELPHER: str = "delpher"

    # Select decade
    DECADE: str = "1980s"
    # Bool to ungizp metadata, do only once!
    UNGIZP: bool = True
    # Decide whether to process and save articles and metadata data
    DATAFILE: Dict = {
        "start": "True",
        "metadata": "True",
        "articles": "True",
    }
    # Merge articles and metadata
    MERGE: bool = True
    # Selected topic
    TOPIC: str = "coal"
    # Run search words and add metadata to articles csv
    SEARCH_WORDS: bool = False
    LIST_INCL_WORDS: Any = kw.KEYWORDS_KOOL
    LIST_EXCL_WORDS: Any = kw.EXCL_WORDS_KOOL
    # Preprocess articles for labeling
    PREPROCESS: bool = False
    # Run classification model
    CLASSIFY: bool = False

    # If other decade than current used to classify with NB
    # set custom labels to True and select the decade below
    CUSTOM_LABELS: bool = True
    CUSTOM_DECADES: List = ["1970s", "1980s", "1990s"]

    class Config:
        case_sensitive = True


settings = Settings()
