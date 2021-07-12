from typing import Any, Dict, List, Optional, Union

from pydantic import BaseSettings

class Settings(BaseSettings):
    # Raw files
    PATH_RAW_FILES: str = "/home/leonardovida/data/volume_1/data-histaware/merged_articles/1970s"
    PATH_DATASET_DIR: str = "/home/leonardovida/data/volume_1/data-histaware/dataset"
    SOURCE_PATH: str = "/home/leonardovida/data/volume_1/data-histaware/merged_articles/1970s/merged_articles/test/2.txt"

    # Tokenizer
    VOCAB_PATH: str = "/home/leonardovida/data/volume_1/delphbert-results/2-tokenizers/1960/dutch.vocab.mod"
    PATH_TOKENIZER_DIR: str = "/home/leonardovida/data/volume_1/delphbert-results/2-tokenizers/1960"

    # Dataset path
    DEST_PATHS: str = "/home/leonardovida/data/volume_1/delphbert-results/3-pretraining_data_cased/1960"
