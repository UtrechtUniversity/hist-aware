# https://www.sbert.net/docs/
from sentence_transformers import SentenceTransformer, LoggingHandler, util

# These are the pure transformers from huggingface
import torch
import pandas as pd
import numpy as np

from src import logger

# Set fixed random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Logger
HAlogger = logger.get_logger("semantic-search")
HAlogger.debug("Test message")

if __name__ == "__main__":
    HAlogger.debug("Import data")
    # Import data
    search = pd.read_csv("./data/processed/selected_articles/2020-10-07_aardgas.csv")

    # Find GPU on device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    HAlogger.debug("Load transformer")
    # embedder = SentenceTransformer(
    #     "./data/models/distiluse/distiluse-base-multilingual-cased"
    # )
    embedder = SentenceTransformer("./data/models/bertje")

    # Corpus
    HAlogger.debug("Creating embeddings for the corpus")
    corpus = list(search["text"])[11:]
    corpus_embeddings = embedder.encode(
        corpus,
        device=device,
        show_progress_bar=True,
        convert_to_tensor=True,
        num_workers=2,
    )

    # Query sentences:
    HAlogger.debug("Creating embeddings for the queries")
    queries = list(search["text"])[0:10]
    queries_embeddings = embedder.encode(
        queries,
        device=device,
        show_progress_bar=True,
        convert_to_tensor=True,
        num_workers=2,
    )

    test = util.semantic_search(
        query_embeddings=queries_embeddings,
        corpus_embeddings=corpus_embeddings,
        query_chunk_size=10,
        corpus_chunk_size=1000,
        top_k=100,
    )

    for a in test:
        print(a)

    # # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = 10
    for query in queries:
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        cos_scores = cos_scores.cpu()

        # We use np.argpartition, to only partially sort the top_k results
        top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

        print("\n\n======================\n\n")
        print("Query:", query)
        print("\nTop 5 most similar sentences in corpus:")

        for idx in top_results[0:top_k]:
            print(corpus[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))