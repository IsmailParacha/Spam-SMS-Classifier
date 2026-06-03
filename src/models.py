from typing import Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def build_models() -> Dict[str, Pipeline]:
    """
    Build multiple machine learning models.
    Each model uses TF-IDF + classifier.
    """

    models = {
        "MultinomialNB": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("model", MultinomialNB())
        ]),

        "LogisticRegression": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ]),

        "LinearSVM": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("model", LinearSVC(class_weight="balanced"))
        ]),

        "DecisionTree": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("model", DecisionTreeClassifier(max_depth=25, random_state=42))
        ])
    }

    return models