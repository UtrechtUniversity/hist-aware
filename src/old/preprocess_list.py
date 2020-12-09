import re, string, unicodedata
from collections import Counter
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import spacy
import nl_core_news_lg

import enchant


class CleanText:
    def __init__(self):
        self.d = enchant.Dict("nl_NL")
        self.nlp = nl_core_news_lg.load()

    def remove_numbers(self):
        self.text = re.sub("[-+]?[0-9]+", "", self.text)
        return self

    def get_words(self):
        self.words = nltk.word_tokenize(self.text)
        return self

    def remove_non_ascii(self):
        """Remove non-ASCII characters from list of tokenized words"""
        new_words = []
        for word in self.words:
            new_word = (
                unicodedata.normalize("NFKD", word)
                .encode("ascii", "ignore")
                .decode("utf-8", "ignore")
            )
            new_words.append(new_word)
        self.words = new_words
        return self

    def to_lowercase(self):
        """Convert all characters to lowercase from list of tokenized words"""
        new_words = []
        for word in self.words:
            new_word = word.lower()
            new_words.append(new_word)
        self.words = new_words
        return self

    def remove_punctuation(self):
        """Remove punctuation from list of tokenized words"""
        new_words = []
        for word in self.words:
            new_word = re.sub(r"[^\w\s]", "", word)
            if new_word != "":
                new_words.append(new_word)
        self.words = new_words
        return self

    def remove_stopwords(self):
        """Remove stop words from list of tokenized words"""
        new_words = []
        for word in self.words:
            if word not in stopwords.words("dutch"):
                new_words.append(word)
        self.words = new_words
        return self

    def remove_frequent(self):
        """Remove frequent words"""
        new_words = []
        for word in self.words:
            if word not in self.FREQWORDS:
                new_words.append(word)
        self.words = new_words
        return self

    def count_frequent(self):
        """Count frequent words"""
        cnt = Counter()
        for word in self.words:
            cnt[word] += 1
        self.FREQWORDS = set([w for (w, wc) in cnt.most_common(100)])
        return self

    def remove_one_char(self):
        new_words = []
        for word in self.words:
            if len(word) > 1:
                new_words.append(word)
        self.words = new_words
        return self

    def remove_non_words(self):
        new_words = []
        for word in self.words:
            if self.d.check(word):
                new_words.append(word)
        self.words = new_words
        return self

    def stem_words(self):
        """Stem words in list of tokenized words"""
        stemmer = SnowballStemmer("dutch")
        stems = []
        for word in self.words:
            stem = stemmer.stem(word)
            stems.append(stem)
        self.words = stems
        return self

    def join_words(self):
        self.words = " ".join(self.words)
        return self

    def do_all(self, text):

        self.text = text
        self = self.remove_numbers()
        self = self.get_words()
        self = self.remove_punctuation()
        self = self.remove_non_ascii()
        self = self.remove_stopwords()
        self = self.count_frequent()
        self = self.count_frequent()
        self = self.remove_frequent()
        self = self.remove_one_char()
        self = self.remove_non_words()
        self = self.join_words()

        return self.words
