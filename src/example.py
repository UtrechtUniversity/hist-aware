# example.py
import sys
import time

from src import preprocess
from src import tfidf

import pandas as pd
from pandarallel import pandarallel

sys.path.insert(0, "..")

ct = preprocess.TextCleaner()

# Load
df = pd.read_csv("./notebooks/test_df.csv")

# Preprocess step

# This step can be done with pandarallel or dask
start_time = time.time()
pandarallel.initialize(progress_bar=True)
# df["text_wop"] = df["text"].parallel_apply(ct.preprocess)
df["text_clean"] = df["text"].apply(ct.preprocess)
print("--- %s seconds ---" % (time.time() - start_time))

# Save step
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
