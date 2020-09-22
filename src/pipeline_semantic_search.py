# https://www.sbert.net/docs/
from sentence_transformers import SentenceTransformer, LoggingHandler, util

# These are the pure transformers from huggingface
import transformers
import torch
from tqdm import tqdm
import hdbscan
import pandas as pd
import numpy as np
import logging
from pylab import rcParams
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import hnswlib

import logger

# Set fixed random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Logger
HAlogger = logger.get_logger("semantic-search")
HAlogger.debug("Test message")

### CODE

if __name__ == '__main__':
    HAlogger.debug("Import data")
    # Import data
    search = pd.read_csv("../data/processed/selected_articles/2020-09-22_energie.csv")

    # Find GPU on device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    HAlogger.debug("Load transformer")
    embedder = SentenceTransformer('../data/models/distiluse-base-multilingual-cased')

    # Corpus
    HAlogger.debug("Creating embeddings for the corpus")
    corpus = list(search["text"])[10:]
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

    # Query sentences:
    HAlogger.debug("Creating embeddings for the queries")
    queries = list(search["text"])[0:10]
    queries_embeddings = embedder.encode(queries, convert_to_tensor=True)

    test = util.semantic_search(
        query_embeddings=queries_embeddings,
        corpus_embeddings=corpus_embeddings,
        query_chunk_size=10,
        corpus_chunk_size=1000,
        top_k=50
    )

    for a in test:
        print(a)

    # # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    # top_k = 10
    # for query in queries:
    #     query_embedding = embedder.encode(query, convert_to_tensor=True)
    #     cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    #     cos_scores = cos_scores.cpu()

    #     #We use np.argpartition, to only partially sort the top_k results
    #     top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

    #     print("\n\n======================\n\n")
    #     print("Query:", query)
    #     print("\nTop 5 most similar sentences in corpus:")

    #     for idx in top_results[0:top_k]:
    #         print(corpus[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))