"""Data loading, cleaning, and TF-IDF feature extraction for the CEAS_08
phishing email dataset.

This is the team's original preprocessing with two correctness fixes:

1. Missing ``subject``/``body`` values are filled *before* concatenation.
   Joining a present subject with a missing body (NaN) would otherwise
   produce NaN and silently discard the subject too.
2. Empty and exact-duplicate emails are removed, and the index is reset so
   the feature matrix ``X`` and labels ``y`` stay aligned.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def load_and_clean(csv_path):
    """Load the CEAS_08 CSV and return a cleaned DataFrame.

    The returned DataFrame has a single ``subject_and_body`` text column and
    the original ``label`` column (1 = phishing, 0 = legitimate).
    """
    df = pd.read_csv(csv_path)

    # Fix 1: fill each column BEFORE combining them.
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    df["subject_and_body"] = (df["subject"] + " " + df["body"]).str.strip()
    df = df.drop(columns=["subject", "body"])

    # Fix 2: drop empty and duplicate emails, then reset the index.
    df = df[df["subject_and_body"] != ""]
    df = df.drop_duplicates(subset=["subject_and_body"]).reset_index(drop=True)
    return df


def build_features(df, max_features=5000):
    """Transform the cleaned text into a sparse TF-IDF matrix.

    Returns ``(X, y, vectorizer)`` where ``X`` is a sparse matrix kept in its
    memory-efficient form (never densified) and ``y`` is the label Series.
    """
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
    X = vectorizer.fit_transform(df["subject_and_body"])
    y = df["label"]
    return X, y, vectorizer
