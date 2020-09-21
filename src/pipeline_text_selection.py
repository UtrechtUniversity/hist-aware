"""
This script contains the pipeline to select the first naive text
selection on the delpher dataset
"""
import pathlib
import sys
import pickle
import csv
import os
import gzip
import sys
import re
import shutil
from datetime import datetime
from pyfiglet import Figlet

from tqdm import tqdm
import hdbscan
import pandas as pd
import numpy as np
import logging
import seaborn
import xml.etree.ElementTree as et 
import collections
import xmltodict
from itertools import chain
import spacy
import nl_core_news_lg

# Import modules
import iterators
import parsers
import logger
import text_selection

#### Just some code to print debug information to stdout
np.set_printoptions(threshold=100)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

# Setup logger
HAlogger = logger.get_logger("pipeline")
HAlogger.debug("Test pipeline")

### VARIABLES ###

# Set Paths
# TODO: fix the MAIN_PATH
MAIN_PATH = sys.path
sys.path.insert(0, "..")
# Save path
SAVE_PATH=MAIN_PATH[0]+"/data/processed/"
# Data path for Delpher data
DIR_PATH="/data/1950/"
# Decide whether to ungizip metadata
UNGIZP=False
# Decide whether to process and save articles and metadata data
DATAFILE=False
# Keywords to use for the naive text selection
KEYWORDS = ["energie"]
# Number of synonyms to retrieve for each keyword, the more the less accurate
NUM_SYNONYMS = 50
# Transformer model to use for the creation of the synonyms
NLP = nl_core_news_lg.load()

# TODO: make the ungizip iterate over the entire data

if __name__ == '__main__':
    f = Figlet(font='slant')
    print(f.renderText('HistAware'))

    # Iterate in the directory and retrieve all the xml article names
    xml_article_names = iterators.iterate_directory(
        root_path=MAIN_PATH[0],
        dir_path=DIR_PATH,
        file_type=".xml")
    article_names = pd.DataFrame.from_dict(xml_article_names).reset_index(inplace=True)

    # If true, ungizp the metadata
    if UNGIZP:
        iterators.ungzip_metdata(
            root_path=MAIN_PATH[0],
            dir_path="/data/1950/",
            file_type=".gz"
        )
    
    # Iterate in the directory and retrieve all the names of the metadata
    gz_metadata_files = iterators.iterate_directory_gz(
        root_path=MAIN_PATH[0],
        dir_path=DIR_PATH,
        file_type=".gz")
    metadata_files = pd.DataFrame.from_dict(gz_metadata_files).reset_index(inplace=True)

    # # Process and save datafiles
    if DATAFILE:
        iterators.iterate_files(save_path=SAVE_PATH, files=article_names)
        iterators.iterate_metadata(save_path=SAVE_PATH, files=metadata_files)

    # Find path and name of saved data
    csv_articles = iterators.iterate_directory(
        root_path=MAIN_PATH[0],
        dir_path=SAVE_PATH+"processed_articles",
        file_type=".csv")
    csv_articles = pd.DataFrame(csv_articles)
    csv_articles.rename({'article_name': 'csv_name', 'article_path': 'csv_path', 'article_dir': 'csv_dir'}, axis=1, inplace=True)

    csv_metadata = iterators.iterate_directory(
        root_path=MAIN_PATH[0],
        dir_path=SAVE_PATH+"processed_metadata",
        file_type=".csv")
    csv_metadata = pd.DataFrame(csv_metadata)
    csv_metadata.rename({'article_name': 'csv_name', 'article_path': 'csv_path', 'article_dir': 'csv_dir'}, axis=1, inplace=True)

    # Iterate on metadata
    for index, row in csv_metadata.iterrows():    
        csv = pd.read_csv(row["csv_path"])
        result = result.append(csv)
    df_metadata = result
    df_metadata.drop(["level_0", "date"], axis=1, inplace=True)
    df_metadata.rename({"filepath": "metadata_filepath", "index": "index_metadata"}, axis=1, inplace=True)
    
    for filename in os.listdir(SAVE_PATH+"processed_articles/"):
        csv = pd.read_csv(row["csv_path"])
        result = result.append(csv)
    
    # Iterate on articles 500.000 articles at the time
    for i, row in csv_articles.iterrows():
        csv = pd.read_csv(row["csv_path"])
        result = result.append(csv)
        if (i % 10 == 0):
            df_articles = result
            df_articles.sort_values(by=["index"], ascending=True)
            df_articles.rename({"filepath": "article_filepath", "index": "index_article"}, axis=1, inplace=True)
            df_joined = df_articles.merge(df_metadata, how='left', on='dir')
            
            result = []

    # Merge in a single dataframe

    for keyword in KEYWORDS:
        selected_art = text_selection.select_articles(
            nlp=NLP,
            word=keyword,
            df=df_joined,
            n=NUM_SYNONYMS
        )
        NAME = str(datetime.date.today())+"_"+keyword
        selected_art.to_csv(SAVE_PATH+NAME,
            sep=",",
            quotechar='"',
            index=False
        )
