import pandas as pd
from datasets import Dataset, DatasetDict
from pathlib import Path
from huggingface_hub import login


def csv_to_huggingface_dataset(
    csv_path: str,
    output_dir: str | None = None,
    dropna_subset: list[str] | None = None,
    columns: list[str] | None = None,
) -> Dataset | DatasetDict:
    df = pd.read_csv(csv_path)
    
    if columns:
        df = df[columns]
    if dropna_subset:
        df = df.dropna(subset=dropna_subset)
    
    dataset = Dataset.from_pandas(df)
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_path))
        print(f" Dataset saved to {output_path}")
        print(f"  Total examples: {len(dataset)}")
        print(f"  Features: {list(dataset.features.keys())}")
    
    return dataset


def csv_to_huggingface_dataset_with_splits(
    csv_path: str,
    output_dir: str,
    test_size: int = 50,
    val_size: int = 30,
    random_state: int = 42,
    dropna_subset: list[str] | None = None,
    stratify_column: str | None = None,
    columns: list[str] | None = None,
) -> DatasetDict:
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv(csv_path)
    
    if columns:
        df = df[columns]
    
    if dropna_subset:
        df = df.dropna(subset=dropna_subset)
    
    if stratify_column:
        y = df[stratify_column]
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, stratify=y, random_state=random_state
        )
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size,
            stratify=train_val_df[stratify_column],
            random_state=random_state,
        )
    else:
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size, random_state=random_state
        )
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset,
    })
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))
    
    print(f"  DatasetDict saved to {output_path}")
    print(f"  Train: {len(train_dataset)} examples")
    print(f"  Validation: {len(val_dataset)} examples")
    print(f"  Test: {len(test_dataset)} examples")
    print(f"  Features: {list(train_dataset.features.keys())}")
    
    return dataset_dict


def upload_to_hub(
    dataset: Dataset | DatasetDict,
    repo_id: str,
    private: bool = False,
    token: str | None = None,
) -> None:
    if token:
        login(token=token)
    
    print(f"\nUploading to Hugging Face Hub: {repo_id}")
    dataset.push_to_hub(repo_id=repo_id, private=private)
    print(f"Successfully uploaded to https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    csv_path = "llm_judge/dataset/primock_data_final_outcomes.csv"
    
    selected_columns = [
        "index",
        "composite_key",
        "doctor",
        "patient_ground_truth",
        "patient_hypothesis",
        "alignment_status",
        "provider",
        "norm_ground_truth",
        "norm_hypothesis",
        "fer_gt_context",
        "fer_hyp_context",
        "final_outcome",
    ]
    
    # Option 1: Convert to single dataset
    print("Converting CSV to Hugging Face dataset...")
    dataset = csv_to_huggingface_dataset(
        csv_path=csv_path,
        output_dir="llm_judge/dataset/primock_hf_dataset",
        dropna_subset=["final_outcome"],
        columns=selected_columns,
    )
    
    # Option 2: Convert with train/val/test splits
    print("\nConverting CSV to Hugging Face DatasetDict with splits...")
    dataset_dict = csv_to_huggingface_dataset_with_splits(
        csv_path=csv_path,
        output_dir="llm_judge/dataset/primock_hf_dataset_splits",
        test_size=50,
        val_size=30,
        random_state=42,
        dropna_subset=["final_outcome"],
        stratify_column="final_outcome",
        columns=selected_columns,
    )
    
    print("\n✓ Conversion complete!")
    
    # Upload to Hugging Face Hub (uncomment and set your username and repo_id)
    repo_id = "username/repo-id"
    upload_to_hub(dataset_dict, repo_id=repo_id, private=False)

