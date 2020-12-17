import os

import pandas as pd

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn import model_selection, preprocessing, metrics
from sklearn.feature_extraction.text import TfidfVectorizer

from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.metrics import classification_report_imbalanced

from loguru import logger


class ClassifyArticles:
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
        self.grid_params = {
            "tfidfvectorizer__analyzer": ["word"],
            "tfidfvectorizer__token_pattern": [r"\w{1,}"],
            "tfidfvectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)],
            "tfidfvectorizer__smooth_idf": [True, False],
            "tfidfvectorizer__sublinear_tf": [1],
            "tfidfvectorizer__strip_accents": ["unicode"],
            "tfidfvectorizer__use_idf": [True, False],
            "tfidfvectorizer__min_df": [1, 2, 3],
            "tfidfvectorizer__max_features": [None, 5000, 10000, 50000],
        }

    def load(self):
        # Read file
        train = pd.read_csv(
            os.path.join(
                self.SELECTED_DECADE, f"{self.DECADE}_{self.TOPIC}_labeled.csv"
            )
        )
        # Drop na rows
        train.dropna(0, subset=["text_clean", "sentiment"], inplace=True)
        # Eliminate titles
        self.train = train[train["type"] != "title"].copy()
        return self.train

    def split(self, data):
        # Spliting into X & y
        # X = train.iloc[:, 2].values # to get th
        X = data["text_clean"].values  #  to get th

        # Convert label to numeric
        cleanup_label = {"sentiment": {"Yes": 1, "No": 0}}
        data.replace(cleanup_label, inplace=True)
        y = data.sentiment.values

        # Split train and validation
        self.train_x, self.valid_x, self.train_y, self.valid_y = train_test_split(
            X, y, stratify=y, random_state=42, test_size=0.2, shuffle=True
        )

        # label encode the target variable
        encoder = preprocessing.LabelEncoder()
        self.train_y = encoder.fit_transform(self.train_y)
        self.valid_y = encoder.fit_transform(self.valid_y)

    def make_pipeline(self, model, sampler, classifier):
        self.pipe = make_pipeline_imb(model, sampler, classifier)
        self.pipe.fit(self.train_x, self.train_y)
        y_pred = self.pipe.predict(self.valid_x)
        print(classification_report_imbalanced(self.valid_y, y_pred))
        return self.pipe

    # Load the previous dataset for testing
    def predict(self, pipe, THRESHOLD):
        test = pd.read_csv(
            f"{self.SELECTED_DECADE}/{self.DECADE}_{self.TOPIC}_to_label.csv"
        )
        test.rename(columns={"label": "sentiment"}, inplace=True)
        test["sentiment"] = None

        # Select only new rows and not the training set
        test_df = test.loc[~test.index_article.isin(self.train.index_article)].copy()

        # Drop na and titles
        test_df.dropna(0, subset=["text_clean"], inplace=True)
        test_df = test_df[test_df["type"] != "title"].copy()

        # Predict new values
        self.x_test = test_df["text_clean"].values
        self.y_test = pipe.predict_proba(self.x_test)

        # Add predicted values to dataframe
        test_df["sentiment"] = self.y_test

        # Select only values below threshold
        self.df_labeled = test_df[test_df["sentiment"] < THRESHOLD]

        # Save labeled df
        self.df_labeled.to_csv(
            f"{self.SELECTED_DECADE}/{self.DECADE}_{self.TOPIC}_labeled_full_{THRESHOLD}.csv"
        )
        logger.info(
            f"Classification results saved as '{self.DECADE}_{self.TOPIC}_labeled_full_{THRESHOLD}.csv'"
        )

    def run_classification_pipeline(self, sampler, classifier, THRESHOLD=0.1):
        logger.debug("Load and split data")
        # Load and split data
        train_data = self.load()
        self.split(train_data)

        # Make pipeline
        logger.debug("Create pipeline")
        pipe = self.make_pipeline(TfidfVectorizer(), sampler, classifier)

        # Search grid
        logger.info("Searching grid")
        grid_search = GridSearchCV(pipe, self.grid_params, cv=10)
        grid_search.fit(self.train_x, self.train_y)

        logger.info("Grid search completed. Best params:")
        for k, v in grid_search.best_params_.items():
            print(k, " : ", v)
        print(grid_search.best_score_)

        # Predict using best estimators
        logger.info("Make pipeline with best results")
        best_res = grid_search.cv_results_["params"][grid_search.best_index_]
        return self.make_pipeline(TfidfVectorizer(input=best_res), sampler, classifier)


# def grid_search(pipe, X_train, X_test, y_train, y_test):
#     """Use multiple classifiers and grid search for prediction"""

#     if not set(pipe.keys()).issubset(set(params.keys())):
#         raise ValueError("Some estimators are missing parameters")

#     for key in pipe.keys():

#         pipe = pipe[key]
#         param = params[key]
#         gs = GridSearchCV(pipe, param, cv=10, error_score=0, refit=True)
#         gs.fit(X_train, y_train)
#         y_pred = gs.predict(X_test)

#         # Print scores for the classifier
#         print(key, ":", gs.best_params_)
#         print(
#             "Accuracy: %1.3f \tPrecision: %1.3f \tRecall: %1.3f \t\tF1: %1.3f\n"
#             % (
#                 accuracy_score(y_test, y_pred),
#                 precision_score(y_test, y_pred, average="macro"),
#                 recall_score(y_test, y_pred, average="macro"),
#                 f1_score(y_test, y_pred, average="macro"),
#             )
#         )

#     return