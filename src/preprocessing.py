import re
import string

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def clean_text(text: str) -> str:
    """
    Clean SMS text.
    Example:
    'Win $1000 now!!!' -> 'win number'
    """

    text = str(text).lower()

    # Replace numbers with word number
    text = re.sub(r"\d+", " number ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Split into words
    tokens = text.split()

    # Remove English stop words
    tokens = [
        token for token in tokens
        if token not in ENGLISH_STOP_WORDS
    ]

    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cleaned message, message length, word count, and numeric label.
    """

    processed = df.copy()

    processed["clean_message"] = processed["message"].apply(clean_text)
    processed["message_length"] = processed["message"].apply(len)
    processed["word_count"] = processed["message"].apply(
        lambda x: len(str(x).split())
    )

    processed["label_num"] = processed["label"].map({
        "ham": 0,
        "spam": 1
    })

    return processed