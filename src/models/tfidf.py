import os

import pandas as pd

from numpy import argmax
from numpy import arange
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.metrics import classification_report_imbalanced

from loguru import logger


class ClassifyArticles:
    """Class used to classify articles."""

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
            "tfidfvectorizer__max_features": [None, 5000, 10000, 50000, 100000],
        }

    def load(self):
        """Read labeled articles."""

        if "all_decades" in self.SELECTED_DECADE:
            logger.info("Using multiple decades to train")
            train_1990s = pd.read_csv(
                f"{self.SELECTED_DECADE}/1990s_{self.TOPIC}_labeled.csv"
            )
            train_1980s = pd.read_csv(
                f"{self.SELECTED_DECADE}/1980s_{self.TOPIC}_labeled.csv"
            )
            train = train_1990s.append(train_1980s)
        else:
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
        """Split train and valid dataset and vectorize them."""

        # Spliting into X & y
        # X = train.iloc[:, 2].values
        X = data["text_clean"].values

        # Convert label to numeric
        cleanup_label = {"sentiment": {"Yes": 1, "No": 0}}
        data.replace(cleanup_label, inplace=True)
        y = data.sentiment.values

        # Split train and validation
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            X, y, stratify=y, random_state=42, test_size=0.2, shuffle=True
        )

        # label encode the target variable
        encoder = preprocessing.LabelEncoder()
        self.train_y = encoder.fit_transform(self.train_y)
        self.test_y = encoder.fit_transform(self.test_y)

    def to_labels(self, pos_probs, threshold):
        """Apply threshold to positive probabilities to create labels."""
        return (pos_probs >= threshold).astype("int")

    def make_pipeline(self, model, sampler, classifier):
        """Create training pipeline."""

        # Create pipeline
        self.pipe = make_pipeline_imb(model, sampler, classifier)
        self.pipe.fit(self.train_x, self.train_y)
        # Predict for report
        y_pred = self.pipe.predict(self.test_x)

        # Search "best" threshold
        y_hat = self.pipe.predict_proba(self.test_x)
        fpr, tpr, thresholds = roc_curve(self.test_y, y_hat[:, 1])
        J = tpr - fpr
        ix = argmax(J)
        best_thresh = thresholds[ix]
        print("Suggested best ROC Threshold=%f" % (best_thresh))

        # calculate roc curves
        precision, recall, thresholds = precision_recall_curve(self.test_y, y_hat[:, 1])
        # convert to f score
        fscore = (2 * precision * recall) / (precision + recall)
        # locate the index of the largest f score
        ix = argmax(fscore)
        print(
            "Suggested best Threshold=%f, F-Score=%.3f" % (thresholds[ix], fscore[ix])
        )

        # define thresholds
        thresholds = arange(0, 1, 0.001)
        # evaluate each threshold
        scores = [
            f1_score(self.test_y, self.to_labels(y_hat[:, 1], t)) for t in thresholds
        ]
        # get best threshold
        ix = argmax(scores)
        print("Threshold=%.3f, F-Score=%.5f" % (thresholds[ix], scores[ix]))

        print(classification_report_imbalanced(self.test_y, y_pred))
        return (self.pipe, thresholds[ix])

    def predict(self, pipe, DECADE_TO_PREDICT, THRESHOLD):
        """Load dataset for predicting.
        One dataset at a time, that can be different that the one
        used to create the model.
        """
        PREDICT_FILE = f"{DECADE_TO_PREDICT}_{self.TOPIC}_to_label.csv"
        PREDICT_PATH = os.path.join(
            self.SAVE_PATH, "labeled_articles", DECADE_TO_PREDICT, PREDICT_FILE
        )
        test = pd.read_csv(PREDICT_PATH)
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

        # Add predicted values (y --> 1) to dataframe
        test_df["sentiment"] = self.y_test[:, 1]

        # Select only values above threshold
        self.df_labeled = test_df[test_df["sentiment"] >= THRESHOLD]

        # Save labeled df
        self.df_labeled.to_csv(
            os.path.join(
                self.SAVE_PATH,
                "labeled_articles",
                DECADE_TO_PREDICT,
                f"{self.DECADE}_{self.TOPIC}_labeled_full_{THRESHOLD}.csv",
            )
        )
        logger.info(
            f"Classification results saved as '{self.DECADE}_{self.TOPIC}_labeled_full_{THRESHOLD}.csv'"
        )

    def run_classification_pipeline(self, sampler, classifier):
        """Main classification pipeline to call previous functions."""

        logger.debug("Load and split data")
        # Load and split data
        train_data = self.load()
        self.split(train_data)

        # Make pipeline
        logger.debug("Create pipeline")
        pipe, thres = self.make_pipeline(TfidfVectorizer(), sampler, classifier)

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
        best_pipe, best_thres = self.make_pipeline(
            TfidfVectorizer(input=best_res), sampler, classifier
        )
        return (best_pipe, best_thres)
