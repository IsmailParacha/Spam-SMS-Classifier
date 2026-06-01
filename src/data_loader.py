import pandas as pd

from src.config import DATASET_PATH


def load_dataset() -> pd.DataFrame:
    """
    Load SMS Spam Collection dataset from data folder.
    Dataset should have two columns: label and message.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            "Please put SMSSpamCollection inside the data folder."
        )

    df = pd.read_csv(
        DATASET_PATH,
        sep="\t",
        header=None,
        names=["label", "message"]
    )

    df = df.dropna()
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    return df