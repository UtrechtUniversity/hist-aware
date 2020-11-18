# pipeline_text_selection.py
"""
This script contains the pipeline to select the first naive text
selection on the delpher dataset
"""
import os
from sys import getsizeof
import csv

from datetime import datetime
from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np
from pyfiglet import Figlet
from tqdm import tqdm
import dask.dataframe as dd
from dask.distributed import Client
from dask.diagnostics import ProgressBar
from multiprocessing.pool import ThreadPool


# Import modules
from iterators import (
    iterate_directory,
    iterate_metadata,
    iterate_files,
    ungzip_metdata,
    iterate_directory_gz,
)
from article_selection import select_articles
from preprocess import TextCleaner

# Just some code to print debug information to stdout
np.set_printoptions(threshold=100)


class TextSearch:
    def __init__(
        self,
        FILE_PATH: Path,
        DIR_PATH: Path,
        SAVE_PATH: Path,
        UNGIZP: bool,
        DATAFILE: dict,
        KEYWORDS: str,
        EXCL_WORDS: str,
        TOPIC: str,
        DECADE: str,
    ) -> None:
        self.FILE_PATH = FILE_PATH
        self.DECADE = DECADE
        self.DIR_PATH = DIR_PATH
        self.SAVE_PATH = SAVE_PATH
        self.UNGIZP = UNGIZP
        self.DATAFILE = DATAFILE
        self.KEYWORDS = KEYWORDS
        self.EXCL_WORDS = EXCL_WORDS
        self.TOPIC = TOPIC

        self.RAW_DECADE = os.path.join(self.DIR_PATH, self.DECADE)
        self.PROC_ART_DECADE = os.path.join(
            self.SAVE_PATH, "processed_articles", self.DECADE
        )
        self.PROC_MET_DECADE = os.path.join(
            self.SAVE_PATH, "processed_metadata", self.DECADE
        )
        self.SELECTED_DECADE = os.path.join(
            self.SAVE_PATH, "selected_articles", self.DECADE
        )
        self.INFO_DECADE = os.path.join(self.SAVE_PATH, "file_info", self.DECADE)

        f = Figlet(font="slant")
        print(f.renderText("HistAware"))

    def write_to_disk(self, file_name, file_data):
        """Save function."""
        file = os.path.join(self.SELECTED_DECADE, f"{file_name}.csv")
        write_mode, header = ("a", False) if os.path.isfile(file) else ("w", True)

        if len(file_data) > 0:
            pd.DataFrame(file_data).to_csv(
                path_or_buf=file,
                index=False,
                mode=write_mode,
                header=header,
                quoting=csv.QUOTE_MINIMAL,
            )

    def ungzip_metadata_files(self) -> None:
        """If true ungizp the metadata files in data/raw"""

        # TODO: make the ungizip iterate over the entire data
        logger.debug("Ungzipping metadata")
        ungzip_metdata(dir_path=self.RAW_DECADE, file_type=".gz")

    # TODO: Create two functions out of this
    def iterate_directories(self) -> None:
        """Iterate directories to catalogue files"""
        if not os.path.isfile(os.path.join(self.INFO_DECADE, "article_info.csv")):
            # Iterate in the directory and retrieve all the xml article names
            logger.debug("Retrieving article information")
            xml_article_names = iterate_directory(
                dir_path=self.RAW_DECADE, file_type=".xml"
            )
            self.article_names = pd.DataFrame.from_dict(xml_article_names)
            self.article_names.reset_index(inplace=True)

            logger.debug(f"Size of directory list {getsizeof(self.article_names)}")
            self.article_names.to_csv(
                os.path.join(self.INFO_DECADE, "article_info.csv")
            )
            logger.debug("Articles list saved")
        else:
            logger.debug("Articles list already exists. Skipping")

        # Iterate in the directory and retrieve all the names of the metadata
        if not os.path.isfile(os.path.join(self.INFO_DECADE, "metadata_info.csv")):
            logger.debug("Retrieving metadata information")
            gz_metadata_files = iterate_directory_gz(
                dir_path=self.RAW_DECADE, file_type=".gz"
            )
            self.metadata_files = pd.DataFrame.from_dict(gz_metadata_files)
            self.metadata_files.reset_index(inplace=True)

            logger.debug(f"Size of metadata list {getsizeof(self.metadata_files)}")
            self.metadata_files.to_csv(
                os.path.join(self.INFO_DECADE, "metadata_info.csv")
            )
            logger.debug("Metadata list saved")
        else:
            logger.debug("Articles list already exists. Skipping")

    def process_metdata(self, save_path, files) -> None:
        if self.DATAFILE["metadata"] == "True":
            logger.debug("Processing and saving metadata to csv of metadata")
            iterate_metadata(save_path, files)
        else:
            logger.debug("Metadata already processed. Skipping.")

    def process_articles(self, save_path, files) -> None:
        if self.DATAFILE["files"] == "True":
            logger.debug("Processing and saving articles to csv of articles")
            iterate_files(save_path, files)
        else:
            logger.debug("Articles already processed. Skipping.")

    def process_files(self) -> None:
        """If DATAFILE is True, then process and save the files.
        This process is extremely time-intensive, so it should be done only once."""

        if self.DATAFILE["start"] == "True":
            # Load and process articles
            self.article_names = pd.read_csv(
                os.path.join(self.INFO_DECADE, "article_info.csv")
            )
            self.process_articles(
                save_path=self.PROC_ART_DECADE, files=self.article_names
            )

            # Load and process metadata
            self.metadata_files = pd.read_csv(
                os.path.join(self.INFO_DECADE, "metadata_info.csv")
            )
            self.process_metdata(
                save_path=self.PROC_MET_DECADE, files=self.metadata_files
            )
        else:
            logger.debug(
                "Skipping processing of both articles and metadata. If you want to \
                change it check the settings."
            )

    def retrieved_saved_files(self) -> None:
        """Retrieve path and name of saved data"""

        logger.debug("Find path and name of saved info about articles (csv)")
        self.csv_articles = iterate_directory(
            dir_path=os.path.join(self.PROC_ART_DECADE),
            file_type=".csv",
        )
        self.csv_articles = pd.DataFrame(self.csv_articles)
        self.csv_articles.rename(
            columns={
                "article_name": "csv_name",
                "article_path": "csv_path",
                "article_dir": "csv_dir",
            },
            inplace=True,
        )

        logger.debug("Find path and name of saved info about metadata (csv)")
        self.csv_metadata = iterate_directory(
            dir_path=os.path.join(self.PROC_MET_DECADE),
            file_type=".csv",
        )
        self.csv_metadata = pd.DataFrame(self.csv_metadata)
        self.csv_metadata.rename(
            columns={
                "article_name": "csv_name",
                "article_path": "csv_path",
                "article_dir": "csv_dir",
            },
            inplace=True,
        )

    def search_synonyms(self) -> None:
        """Using the processed and saved data, search the synonyms"""
        li = []

        # Read all the metadata into one file
        for index, row in self.csv_metadata.iterrows():
            csv_file = pd.read_csv(row["csv_path"])
            li.append(csv_file)

        self.df_metadata = pd.concat(li, axis=0)
        # print(self.df_metadata.columns.values)
        self.df_metadata.drop(columns=["date"], inplace=True)
        self.df_metadata.rename(
            columns={"filepath": "metadata_filepath", "index": "index_metadata"},
            inplace=True,
        )

        # Search synonyms in saved articles
        li = []
        logger.info("Searching keywords")
        # Load processed articles iteratively
        for i, row in tqdm(
            self.csv_articles.iterrows(), total=self.csv_articles.shape[0]
        ):
            csv_file = pd.read_csv(row["csv_path"])
            li.append(csv_file)
            if i % 20 == 0:
                logger.debug(f"Currently parsed {i*50000} articles")
                df_articles = pd.concat(li, axis=0)
                df_articles.sort_values(by=["index"], ascending=True)
                df_articles.rename(
                    columns={"filepath": "article_filepath", "index": "index_article"},
                    inplace=True,
                )
                df_joined = df_articles.merge(self.df_metadata, how="left", on="dir")

                # Search for keywords in loaded csvs
                selected_art = select_articles(
                    words=self.KEYWORDS,
                    excl_words=self.EXCL_WORDS,
                    df=df_joined,
                )
                today = datetime.now()
                NAME = str(today.date()) + "_" + self.TOPIC + "_" + str(i)
                self.write_to_disk(file_name=NAME, file_data=selected_art)

                # Reset for next iteration
                selected_art = []
                df_articles = pd.DataFrame
                li = []

    def process_selected_articles(self):
        tqdm.pandas()
        client = Client()
        csv_temp = []

        # Create preprocessing class
        self.tc = TextCleaner()

        # Load selected articles for selected topic in nlp_pipeline
        csv = iterate_directory(self.SELECTED_DECADE, ".csv")
        [csv_temp.append(c) for c in csv if self.TOPIC in c["article_name"]]

        df = pd.concat(
            [pd.read_csv(c["article_path"]) for c in csv_temp], ignore_index=True
        )
        # Initial clean
        df.drop_duplicates(subset=["text"], inplace=True)
        df.sort_values(by=["count"], ascending=False, inplace=True)
        df.reset_index(inplace=True)
        df.drop(columns={"index", "Unnamed: 0_x", "Unnamed: 0_y"}, inplace=True)

        # Preprocess text to text_clean
        data = df
        ddata = dd.from_pandas(data, npartitions=15)
        ProgressBar().register()

        def apply_preprocess_to_DF(df):
            return df["text"].apply(self.tc.preprocess)

        res = ddata.map_partitions(apply_preprocess_to_DF).compute(scheduler=client)
        # df["text_clean"] = df["text"].progress_apply(self.tc.preprocess)

        # Add label column for labeling
        df.insert(1, "label", "")
        df["text_clean"] = res
        df_name = "to_label_" + str(self.TOPIC) + ".csv"

        # Save
        df.to_csv(os.path.join(self.SELECTED_DECADE, df_name))
