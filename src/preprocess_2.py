cnt = Counter()
for text in df["text_wop_clean"].values:
    for word in text.split():
        cnt[word] += 1

d = enchant.Dict("nl_NL")
FREQWORDS = set([w for (w, wc) in cnt.most_common(20)])


def remove_punctuation(text):
    """Remove punctuation"""
    return "".join([c for c in text if c not in punctuation])


def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


def remove_freqwords(text, FREQWORDS):
    """custom function to remove the frequent words"""
    return " ".join([word for word in str(text).split() if word not in FREQWORDS])


def remove_numeric(text):
    """Remove numbers"""
    return "".join([c for c in text if not c.isdigit()])


def remove_non_ascii(text):
    """Remove non ASCII chars"""
    return "".join([re.sub(r"[^\x00-\x7f]", r" ", c) for c in text])


def remove_extra_whitespace_tabs(text):
    """Remove extra whitespaces and tabs"""
    return re.sub(r"^\s*|\s\s*", " ", text).strip()


def remove_one_char(text):
    return " ".join([w for w in text.split() if len(w) > 1])


def remove_non_words(text):
    """custom function to remove the rare words"""
    return " ".join([word for word in str(text).split() if d.check(word)])


df["text_lower"] = df["text"].str.lower()
df["text_wop"] = df["text_lower"].apply(lambda x: remove_punctuation(x))
df["text_wop_clean"] = df["text_wop"].apply(lambda x: remove_stopwords(x))
df["text_wop_clean"] = df["text_wop_clean"].apply(
    lambda text: remove_freqwords(text, FREQWORDS)
)
df["text_wop_clean"] = df["text_wop_clean"].apply(lambda x: remove_numeric(x))
df["text_wop_clean"] = df["text_wop_clean"].apply(lambda x: remove_non_ascii(x))
df["text_wop_clean"] = df["text_wop_clean"].apply(
    lambda x: remove_extra_whitespace_tabs(x)
)
df["text_wop_clean"] = df["text_wop_clean"].apply(lambda x: remove_one_char(x))
df["text_wop_clean"] = df["text_wop_clean"].apply(lambda text: remove_non_words(text))