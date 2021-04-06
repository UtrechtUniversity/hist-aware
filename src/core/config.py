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

    # Bool to ungizp metadata, do only once!
    UNGIZP: bool = True
    # Decide whether to process and save articles and metadata data
    DATAFILE: Dict = {
        "start": "True",
        "metadata": "True",
        "articles": "True",
    }
    # Run search words and add metadata to articles csv
    SEARCH_WORDS: bool = False
    LIST_INCL_WORDS: Any = kw.KEYWORDS_GAS
    LIST_EXCL_WORDS: Any = kw.EXCL_WORDS_GAS
    # Preprocess articles for labeling
    PREPROCESS: bool = True
    # Run classification model
    CLASSIFY: bool = False

    # Selected topic and decade
    TOPIC: str = "gas"
    DECADE: str = "1960s"

    # If other decade than current used to classify with NB
    # set custom labels to True and select the decade below
    CUSTOM_LABELS: bool = True
    CUSTOM_DECADES: List = ["1970s", "1980s", "1990s"]

    class Config:
        case_sensitive = True


settings = Settings()
