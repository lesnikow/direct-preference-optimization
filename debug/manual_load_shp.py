import os
import json
import datasets
import requests
from pathlib import Path
from typing import Optional, Dict, Any
from huggingface_hub import hf_hub_download


class ManualSHPLoader:
    """Manual loader for Stanford Human Preferences dataset"""

    REPO_ID = "stanfordnlp/SHP"
    DATA_FILES = {
        "train": "train/00000.parquet",
        "validation": "validation/00000.parquet",
        "test": "test/00000.parquet",
    }

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface", "datasets", "shp"
        )
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_split(self, split: str) -> datasets.Dataset:
        """Load a specific split of the dataset"""
        if split not in self.DATA_FILES:
            raise ValueError(
                f"Invalid split: {split}. Must be one of {list(self.DATA_FILES.keys())}"
            )

        try:
            # Download the parquet file using the Hugging Face Hub utility
            filepath = hf_hub_download(
                repo_id=self.REPO_ID,
                filename=self.DATA_FILES[split],
                repo_type="dataset",
                cache_dir=self.cache_dir,
            )

            # Load the parquet file directly into a Dataset object
            return datasets.Dataset.from_parquet(filepath)

        except Exception as e:
            raise Exception(f"Failed to load {split} split: {str(e)}")


def load_shp_dataset(
    split: str = "train", cache_dir: Optional[str] = None
) -> datasets.Dataset:
    """
    Main function to load the SHP dataset

    Args:
        split (str): Dataset split to load ("train", "validation", "test")
        cache_dir (str, optional): Custom cache directory path

    Returns:
        datasets.Dataset: Loaded dataset
    """
    loader = ManualSHPLoader(cache_dir)
    return loader.load_split(split)


# Example usage
if __name__ == "__main__":
    try:
        # Make sure huggingface_hub is installed
        try:
            import huggingface_hub
        except ImportError:
            raise ImportError(
                "Please install huggingface_hub: pip install huggingface_hub"
            )

        dataset = load_shp_dataset("train")
        print(f"Successfully loaded dataset with {len(dataset)} examples")
        print("\nFeatures:", dataset.features)
        print("\nFirst example:")
        print(dataset[0])
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
