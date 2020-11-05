from collections import Counter
from os import sep
from string import punctuation
import re
import pandas as pd

import enchant
from pandarallel import pandarallel
import nltk


class TextCleaner:
    def __init__(self):
        self.d = enchant.Dict("nl_NL")
        self.stopword_list = nltk.corpus.stopwords.words("dutch")
        self.STOPWORDS = set(self.stopword_list)

    def get_words(self):
        self.text = " ".join([c for c in nltk.word_tokenize(self.text)])
        return self

    def lower(self):
        """Lower case the text"""
        self.text = "".join([t.lower() for t in self.text])
        return self

    def remove_stopwords(self):
        """custom function to remove the stopwords"""
        self.text = "".join([t for t in self.text if t not in self.STOPWORDS])
        return self

    def remove_numeric(self):
        """Remove numbers"""
        self.text = "".join([c for c in self.text if not c.isdigit()])
        return self

    def remove_non_ascii(self):
        """Remove non ASCII chars"""
        self.text = "".join([re.sub(r"[^\x00-\x7f]", r" ", c) for c in self.text])
        return self

    def remove_extra_whitespace_tabs(self):
        """Remove extra whitespaces and tabs"""
        self.text = re.sub(r"^\s*|\s\s*", " ", self.text).strip()
        return self

    def remove_one_char(self):
        self.text = " ".join([w for w in self.text.split() if len(w) > 1])
        return self

    def remove_non_words(self):
        """custom function to remove the rare words"""
        self.text = " ".join(
            [word for word in str(self.text).split() if self.d.check(word)]
        )
        return self

    def preprocess(self, text):
        self.text = text
        self = self.get_words()
        self = self.lower()
        self = self.remove_stopwords()
        self = self.remove_numeric()
        self = self.remove_non_ascii()
        self = self.remove_extra_whitespace_tabs()
        self = self.remove_one_char()
        self = self.remove_non_words()
        return self.text
