from typing import Any, Dict, List, Optional, Union

from pydantic import BaseSettings

class Settings(BaseSettings):
    # Raw files
    PATH_RAW_FILES: str = "/home/leonardovida/data/volume_1/data-histaware/merged_articles/1970s"
    PATH_DATASET_DIR: str = "/home/leonardovida/data/volume_1/data-histaware/dataset"
    SOURCE_PATH: str = "/home/leonardovida/data/volume_1/data-histaware/merged_articles/1970s/merged_articles/test/2.txt"

    # Tokenizer
    VOCAB_PATH: str = "~/data/volume_1/data-histaware/tokenizer/dutch.vocab_2"
    PATH_TOKENIZER_DIR: str = "/home/leonardovida/data/volume_1/data-histaware/tokenizer"

    # Dataset path
    DEST_PATHS: str = "/home/leonardovida/data/volume_1/data-histaware/pretrain-data"
