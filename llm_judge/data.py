from typing import List, Tuple

import dspy
import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load dataset and drop rows without final_outcome."""
    df = pd.read_csv(csv_path)
    return df.dropna(subset=["final_outcome"])


def split_dataset(
    df: pd.DataFrame,
    test_size: int = 50,
    val_size: int = 30,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train/val/test with fixed sizes and stratification."""
    y = df["final_outcome"]
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=y, random_state=random_state
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        stratify=train_val_df["final_outcome"],
        random_state=random_state,
    )
    return train_df, val_df, test_df


def create_dspy_example(row) -> dspy.Example:
    """Convert dataframe row to DSPy Example."""
    return dspy.Example(
        ground_truth_conversation=str(row["fer_gt_context"]),
        transcription_conversation=str(row["fer_hyp_context"]),
        clinical_impact=str(int(row["final_outcome"])),
    ).with_inputs("ground_truth_conversation", "transcription_conversation")


def build_splits(
    df: pd.DataFrame, test_size: int = 50, val_size: int = 30, random_state: int = 42
) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
    """Create DSPy train/val/test example lists from dataframe splits."""
    train_df, val_df, test_df = split_dataset(df, test_size, val_size, random_state)
    trainset = [create_dspy_example(row) for _, row in train_df.iterrows()]
    valset = [create_dspy_example(row) for _, row in val_df.iterrows()]
    testset = [create_dspy_example(row) for _, row in test_df.iterrows()]
    return trainset, valset, testset
