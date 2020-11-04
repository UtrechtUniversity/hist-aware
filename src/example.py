import io
from collections import defaultdict
from string import punctuation
import os, os.path
import re
import sys

sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import spacy
from spacy.lemmatizer import Lemmatizer
import nl_core_news_lg
from spacy.lang.nl.stop_words import STOP_WORDS
from tqdm import tqdm_notebook as tqdm
from pprint import pprint
import nltk

from src import iterators

from src import preprocess

# import dask.dataframe as dd
# from dask.multiprocessing import get
import time

df = pd.read_csv("./notebooks/test_df")

start_time = time.time()
ct = preprocess.CleanText()

a = df["text"].apply(ct.do_all)
print("--- %s seconds ---" % (time.time() - start_time))

print(a)
# def dask_this(df):
#     res = df["docs"].apply(ct.do_all)
#     return res


# ddata = dd.from_pandas(df, npartitions=10)

# res = ddata.map_partitions(dask_this).compute(scheduler="processes", num_workers=10)