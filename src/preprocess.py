# preprocess.py
import re

import enchant
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
        """Transform to lower case."""
        self.text = "".join([t.lower() for t in self.text])
        return self

    def remove_stopwords(self):
        """Remove the stopwords."""
        self.text = "".join([t for t in self.text if t not in self.STOPWORDS])
        return self

    def remove_numeric(self):
        """Remove numbers."""
        self.text = "".join([c for c in self.text if not c.isdigit()])
        return self

    def remove_non_ascii(self):
        """Remove non ASCII chars."""
        self.text = "".join([re.sub(r"[^\x00-\x7f]", r" ", c) for c in self.text])
        return self

    def remove_extra_whitespace_tabs(self):
        """Remove extra whitespaces and tabs."""
        self.text = re.sub(r"^\s*|\s\s*", " ", self.text).strip()
        return self

    def remove_one_char(self):
        self.text = " ".join([w for w in self.text.split() if len(w) > 1])
        return self

    def remove_non_words(self):
        """Remove rare words."""
        self.text = " ".join(
            [word for word in str(self.text).split() if self.d.check(word)]
        )
        return self

    def keep_standard_chars(self):
        self.text = "".join([re.sub(r"[^-0-9\w,. ?!()%/]", r"", c) for c in self.text])
        return self

    def preprocess(self, text):
        self.text = text
        self = self.get_words()
        self = self.lower()
        self = self.remove_stopwords()
        self = self.remove_numeric()
        self = self.remove_extra_whitespace_tabs()
        self = self.remove_one_char()
        self = self.remove_non_words()
        return self.text

    def clean(self, text):
        self.text = text
        self = self.get_words()
        self = self.keep_standard_chars()
        self = self.remove_extra_whitespace_tabs()
        return self.text
