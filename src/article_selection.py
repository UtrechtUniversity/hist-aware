# text_selection.py
import re


def select_articles(words, excl_words, df):
    res = search_synonyms(words, excl_words, df)
    # Drop duplicates to keep only individual articles
    # but sum the "count" column
    # res.groupby(
    #     by=[
    #         "Unnamed: 0_x",
    #         "type",
    #         "text",
    #         "article_name",
    #         "date",
    #         "index_article",
    #         "article_filepath",
    #         "dir",
    #         "Unnamed: 0_y",
    #         "metadata_title",
    #         "index_metadata",
    #         "metadata_filepath",
    #         "newspaper_title",
    #         "newspaper_date",
    #         "newspaper_city",
    #         "newspaper_publisher",
    #         "newspaper_source",
    #         "newspaper_volume",
    #         "newspaper_issuenumber",
    #         "newspaper_language",
    #     ]
    # )["count"].sum().reset_index()
    # Sort by counts
    res.sort_values(by=["count"], ascending=False, inplace=True)
    return res


def search_synonyms(words, excl_words, df):
    """Find all texts in which a synonym of the word appears.

    Takes:
        - string (word)
        - dataframe in which to search
        - The total number of synonym to retrieve
    """
    # appended_data = []

    # Create multiple whole word search and exclusion regex
    w = r"\b(?:{})\b".format("|".join(map(re.escape, words)))
    excl_w = r"\b(?:{})\b".format("|".join(map(re.escape, excl_words)))

    # Drops na rows
    df.dropna(subset=["p"], inplace=True)
    # Searches keywords
    res = df[df["p"].str.contains(w, case=False, na=False)].copy()
    # Excludes keywords
    res = res[~res["p"].str.contains(excl_w, case=False, na=False)].copy()
    res["count"] = res["p"].str.count(w, re.I)

    # appended_data.append(res)
    # appended_df = pd.concat(appended_data)

    return res
