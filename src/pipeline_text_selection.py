"""
This script contains the pipeline to select the first naive text
selection on the delpher dataset
"""
import sys
import csv
import os

from tqdm import tqdm
from datetime import datetime
import pandas as pd
import numpy as np
import logging
import nl_core_news_lg
from pyfiglet import Figlet

# Import modules
import iterators
import parsers
import logger
import text_selection

#### Just some code to print debug information to stdout
np.set_printoptions(threshold=100)

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)

# Setup logger
HAlogger = logger.get_logger("pipeline")
HAlogger.debug("Test message")

### VARIABLES ###

# Set Paths
# TODO: fix the PATHS: SAVE, DIR and assignment
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, "..")
# Save path
SAVE_PATH = "../data/processed/"
# Data path for Delpher data
DIR_PATH = "/data/1950/"
# Decide whether to ungizip metadata
UNGIZP = False
# Decide whether to process and save articles and metadata data
DATAFILE = False
# Keywords to use for the naive text selection
KEYWORDS = ["energie"]
# Number of synonyms to retrieve for each keyword, the more the less accurate
NUM_SYNONYMS = 50
# Transformer model to use for the creation of the synonyms
NLP = nl_core_news_lg.load()

# TODO: make the ungizip iterate over the entire data

if __name__ == "__main__":
    f = Figlet(font="slant")
    print(f.renderText("HistAware"))

    # Iterate in the directory and retrieve all the xml article names
    HAlogger.debug("Retrieving article information")
    xml_article_names = iterators.iterate_directory(dir_path=DIR_PATH, file_type=".xml")
    article_names = pd.DataFrame.from_dict(xml_article_names)
    article_names.reset_index(inplace=True)

    # If true, ungizp the metadata
    if UNGIZP:
        HAlogger.debug("Ungzipping metadata")
        iterators.ungzip_metdata(dir_path="/data/1950/", file_type=".gz")

    # Iterate in the directory and retrieve all the names of the metadata
    HAlogger.debug("Retrieving metadata information")
    gz_metadata_files = iterators.iterate_directory_gz(
        dir_path=DIR_PATH, file_type=".gz"
    )
    metadata_files = pd.DataFrame.from_dict(gz_metadata_files)
    metadata_files.reset_index(inplace=True)

    # # Process and save datafiles
    if DATAFILE:
        HAlogger.debug("Processing and saving metadata to csv files")
        iterators.iterate_metadata(save_path=SAVE_PATH, files=metadata_files)
        HAlogger.debug("Processing and saving articles to csv files")
        iterators.iterate_files(save_path=SAVE_PATH, files=article_names)
    else:
        HAlogger.debug("Skipping processing and saving to csv files")

    # Find path and name of saved data
    HAlogger.debug("Find path and name of saved articles")
    csv_articles = iterators.iterate_directory(
        dir_path="/data/processed/processed_articles", file_type=".csv"
    )
    csv_articles = pd.DataFrame(csv_articles)
    csv_articles.rename(
        {
            "article_name": "csv_name",
            "article_path": "csv_path",
            "article_dir": "csv_dir",
        },
        axis=1,
        inplace=True,
    )

    HAlogger.debug("Find path and name of saved metadata")
    csv_metadata = iterators.iterate_directory(
        dir_path="/data/processed/processed_metadata", file_type=".csv"
    )
    csv_metadata = pd.DataFrame(csv_metadata)
    csv_metadata.rename(
        {
            "article_name": "csv_name",
            "article_path": "csv_path",
            "article_dir": "csv_dir",
        },
        axis=1,
        inplace=True,
    )

    # Iterate on metadata
    li = []
    for index, row in csv_metadata.iterrows():
        csv_file = pd.read_csv(row["csv_path"])
        li.append(csv_file)
    df_metadata = pd.concat(li, axis=0)
    df_metadata.drop(["level_0", "date"], axis=1, inplace=True)
    df_metadata.rename(
        {"filepath": "metadata_filepath", "index": "index_metadata"},
        axis=1,
        inplace=True,
    )

    # Search synonyms in saved articles
    li = []
    HAlogger.debug("Searching synonyms")
    for i, row in csv_articles.iterrows():
        csv_file = pd.read_csv(row["csv_path"])
        li.append(csv_file)
        if i % 5 == 0:
            # Iterate 250.000 articles at the time
            df_articles = pd.concat(li, axis=0)
            df_articles.sort_values(by=["index"], ascending=True)
            df_articles.rename(
                {"filepath": "article_filepath", "index": "index_article"},
                axis=1,
                inplace=True,
            )
            df_joined = df_articles.merge(df_metadata, how="left", on="dir")

            for keyword in KEYWORDS:
                HAlogger.debug(f"Searching synonym {keyword}")
                selected_art = text_selection.select_articles(
                    nlp=NLP, word=keyword, df=df_joined, n=NUM_SYNONYMS
                )
                NAME = str(datetime.date.today()) + "_" + keyword + ".csv"

                selected_art.to_csv(
                    "../data/processed/selected_articles/" + NAME,
                    sep=",",
                    quotechar='"',
                    index=False,
                )

            # Reset list of saved csv to zero
            articles_csv_files = []
