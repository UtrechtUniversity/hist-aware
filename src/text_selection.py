# text_selection.py
from tqdm import tqdm
import pandas as pd
import numpy as np
import re


def select_articles(nlp, word, df, n):
    res = search_synonyms(nlp, word, df, n)
    # Drop duplicates to keep only individual articles
    # but sum the "count" column
    res.groupby(
        by=[
            "Unnamed: 0_x",
            "type",
            "text",
            "article_name",
            "date",
            "index_article",
            "article_filepath",
            "dir",
            "Unnamed: 0_y",
            "metadata_title",
            "index_metadata",
            "metadata_filepath",
            "newspaper_title",
            "newspaper_date",
            "newspaper_city",
            "newspaper_publisher",
            "newspaper_source",
            "newspaper_volume",
            "newspaper_issuenumber",
            "newspaper_language",
        ]
    )["count"].sum().reset_index()
    # Sort by counts
    res.sort_values(by=["count"], ascending=False, inplace=True)
    return res


def search_synonyms(nlp, word, df, n):
    """Find all texts in which a synonym of the word appears.

    Takes:
        - string (word)
        - dataframe in which to search
        - The total number of synonym to retrieve
    """
    appended_data = []

    ms = nlp.vocab.vectors.most_similar(
        np.asarray([nlp.vocab.vectors[nlp.vocab.strings[word]]]), n=n
    )
    synonyms = [nlp.vocab.strings[w] for w in ms[0][0]]
    print(f"Searching using the following synonyms of {word}:")
    print(synonyms)
    df.dropna(subset=["text"], inplace=True)

    for syn in tqdm(synonyms):
        # Searches synonym
        res = df[df["text"].str.contains(syn, case=False, regex=False)].copy()
        res["count"] = res["text"].str.count(syn, re.I)
        appended_data.append(res)
    appended_df = pd.concat(appended_data)
    return appended_df
