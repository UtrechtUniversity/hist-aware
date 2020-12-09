import re
import sys
import os

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn import model_selection, preprocessing, metrics, linear_model, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from imblearn.under_sampling import (
    RandomUnderSampler,
    NearMiss,
    InstanceHardnessThreshold,
    CondensedNearestNeighbour,
    EditedNearestNeighbours,
    RepeatedEditedNearestNeighbours,
    AllKNN,
    NeighbourhoodCleaningRule,
    OneSidedSelection,
    TomekLinks,
)
from imblearn.over_sampling import (
    BorderlineSMOTE,
    SMOTE,
    ADASYN,
    SMOTENC,
    RandomOverSampler,
)
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.metrics import classification_report_imbalanced

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import qgrid


class Classification:
    def __init__(
        self,
        SAVE_PATH,
        TOPIC: str,
        DECADE: str,
    ) -> None:
        self.SAVE_PATH = SAVE_PATH
        self.DECADE = DECADE
        self.TOPIC = TOPIC
        self.SELECTED_DECADE = os.path.join(
            self.SAVE_PATH, "labeled_articles", self.DECADE
        )

    def load(self):
        # Read file
        train = pd.read_csv(
            os.path.join(self.SELECTED_DECADE,f"{self.DECADE}_{sel.fTYPE}_labeled.csv")
        # Drop na rows
        train.dropna(0, subset=["text_clean", "sentiment"], inplace=True)
        # Eliminate titles
        train = train[train["type"] != "title"].copy()

    def split(self):
        # Spliting into X & y
        #X = train.iloc[:, 2].values # to get th
        X = train["text_clean"].values # to get th

        # Convert label to numeric
        cleanup_label = {"sentiment": {"Yes": 1, "No": 0}}
        train.replace(cleanup_label, inplace=True)
        y = train.sentiment.values

        # Split train and validation
        from sklearn.model_selection import train_test_split
        train_x, valid_x, train_y, valid_y = train_test_split(X, y, 
                                                        stratify=y, 
                                                        random_state=42, 
                                                        test_size=0.1, shuffle=True)

        # label encode the target variable 
        encoder = preprocessing.LabelEncoder()
        train_y = encoder.fit_transform(train_y)
        valid_y = encoder.fit_transform(valid_y)

    def make_pipeline(self, model, sampler, classifier):
        pipe = make_pipeline_imb(
            model,
            sampler,
            classifier)
        pipe.fit(train_x, train_y)
        return(pipe)



def vectorize(vec, X_train, X_test):
    """Fit the train and test datasets and vectorize"""

    # Fit and transform
    X_train_vec = vec.fit_transform(X_train)
    # Only transform
    X_test_vec = vec.transform(X_test)

    print("Vectorization complete.\n")

    return X_train_vec, X_test_vec


def grid_search(pipe, params, X_train, X_test, y_train, y_test):
    """Use multiple classifiers and grid search for prediction"""

    if not set(pipe.keys()).issubset(set(params.keys())):
        raise ValueError("Some estimators are missing parameters")

    for key in pipe.keys():

        pipe = pipe[key]
        param = params[key]
        gs = GridSearchCV(pipe, param, cv=10, error_score=0, refit=True)
        gs.fit(X_train, y_train)
        y_pred = gs.predict(X_test)

        # Print scores for the classifier
        print(key, ":", gs.best_params_)
        print(
            "Accuracy: %1.3f \tPrecision: %1.3f \tRecall: %1.3f \t\tF1: %1.3f\n"
            % (
                accuracy_score(y_test, y_pred),
                precision_score(y_test, y_pred, average="macro"),
                recall_score(y_test, y_pred, average="macro"),
                f1_score(y_test, y_pred, average="macro"),
            )
        )

    return


if __name__ == "__main__":
param_grid = {
    'tfidfvectorizer__analyzer': ['word'],
    'tfidfvectorizer__token_pattern': [r'\w{1,}'],
    'tfidfvectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],  
    'tfidfvectorizer__smooth_idf': [True, False],
    'tfidfvectorizer__sublinear_tf': [1],
    'tfidfvectorizer__strip_accents': ['unicode'],
    'tfidfvectorizer__use_idf': [True, False],
    'tfidfvectorizer__min_df': [1, 2, 3],
    'tfidfvectorizer__max_features': [None, 5000, 10000, 50000],
}

pipe = make_pipeline(TfidfVectorizer(), SMOTE(), MultinomialNB())

grid_search(pipe)

grid_search = GridSearchCV(pipe, param_grid, cv=10)

#print("Performing grid search...")
#print("pipeline:", [name for name, _ in pipeline.steps])
#print("parameters:")
#pprint(parameters)
#t0 = time()

grid_search.fit(train_x, train_y)