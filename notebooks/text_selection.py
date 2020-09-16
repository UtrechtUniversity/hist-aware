import spacy
from tqdm import tqdm
import hdbscan
import pandas as pd
import numpy as np

def select_articles(nlp, word, df, n):
    res = search_synonyms(nlp, word, df, n)

    # Drop duplicates to keep only individual articles
    res.drop_duplicates(ignore_index=True, inplace=True)

    return(res)

def search_synonyms(nlp, word, df, n):
    """Find all texts in which a synonym of the word appears.
    
    Takes:
        - string (word)
        - dataframe in which to search
        - The total number of synonym to retrieve
    """
    result = pd.DataFrame()
    
    ms = nlp.vocab.vectors.most_similar(np.asarray([nlp.vocab.vectors[nlp.vocab.strings[word]]]), n=n)
    synonyms = [nlp.vocab.strings[w] for w in ms[0][0]]
    print(f"Searching using the following synonyms of {word}:")
    print(synonyms)
    df.dropna(subset=['text'], inplace=True)
    
    for syn in tqdm(synonyms):
        result = result.append(df[df["text"].str.contains(syn, 
                                                          case=False,
                                                          regex=False)
                                 ]
                              )
    return result

### Functions not used

def list_paragraphs(df):
    list_p = []

    for index, row in tqdm(df.iterrows()):
        for i in range(1,df.shape[1]):
            p = "p_"+str(i)
            try:
                if row[p] and row[p] is not None:
                    list_p.append(row[p])
            except KeyError as e:
                continue

    return(list_p)

def list_title(df):
    list_titles = []

    for index, row in tqdm(df.iterrows()):
        try:
            if row["title"] and row["title"] is not None:
                list_titles.append(row["title"])
        except KeyError as e:
            continue

    return(list_titles)