from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def vectorize(vec, X_train, X_test):
    """Fit the traina adn test datasets and vectorize"""

    # Fit and transform
    X_train_vec = vec.fit_transform(X_train)
    # Only transform
    X_test_vec = vec.transform(X_test)

    print("Vectorization complete.\n")

    return X_train_vec, X_test_vec


def grid_search(models, params, X_train, X_test, y_train, y_test):
    """Use multiple classifiers and grid search for prediction"""

    if not set(models.keys()).issubset(set(params.keys())):
        raise ValueError("Some estimators are missing parameters")

    for key in models.keys():

        model = models[key]
        param = params[key]
        gs = GridSearchCV(model, param, cv=10, error_score=0, refit=True)
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
