# example.py
import sys
import time
import os

from src import preprocess
from src import tfidf
from src import iterators

import pandas as pd
from pandarallel import pandarallel

TOPIC = "kool"
DECADE = "1990s"

sys.path.insert(0, "..")

tc = preprocess.TextCleaner()

# Load
csv = iterators.iterate_directory(
    os.path.join("../data/processed/selected_articles/", DECADE), ".csv"
)
df = pd.concat(
    [pd.read_csv(c["article_path"]) for c in csv],
    ignore_index=True,
)
df.drop_duplicates(subset=["text"], inplace=True)
df.sort_values(by=["count"], ascending=False, inplace=True)
df.reset_index(inplace=True)
df.drop(columns={"index", "Unnamed: 0_x", "Unnamed: 0_y"}, inplace=True)

# Preprocess step

# This step can be done with pandarallel or dask
start_time = time.time()
pandarallel.initialize(progress_bar=True)
# df["text_wop"] = df["text"].parallel_apply(tc.preprocess)
df["text_clean"] = df["text"].apply_progress(tc.preprocess)
print("--- %s seconds ---" % (time.time() - start_time))

# Save step
SAVE_PATH
df.to_csv("./notebooks/to_label_final_2.csv", sep=",")

#  TFIDF
# Encode label categories to numbers
# enc = LabelEncoder()
# df = pd.read_csv("to_label_final.csv", sep=";")
# df["label"] = enc.fit_transform(df["label"])
# labels = list(enc.classes_)

# # Train-test split and vectorize
# X_train, X_test, y_train, y_test = train_test_split(
#     df["text_wop"], df["label"], test_size=0.2, shuffle=True
# )
# X_train_vec, X_test_vec = tfidf.vectorize(TfidfVectorizer(), X_train, X_test)

# Preparing to make a pipeline
# models = {
#     "Naive Bayes": MultinomialNB(),
#     "Gradient Boosting": GradientBoostingClassifier(),
# }

# params = {
#     "Naive Bayes": {"alpha": [0.5, 1], "fit_prior": [True, False]},
#     "Gradient Boosting": {"learning_rate": [0.05, 0.1], "min_samples_split": [2, 5]},
# }

# tfidf.grid_search(models, params, X_train_vec, X_test_vec, y_train, y_test)
