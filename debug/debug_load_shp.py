import datasets
from pathlib import Path
import os


def debug_dataset_loading(dataset_name, split="train", cache_dir=None):
    """
    Debug helper for loading datasets from Hugging Face hub with detailed error checking
    and workarounds for common issues.

    Args:
        dataset_name (str): Name of the dataset on HF hub (e.g., "stanfordnlp/SHP")
        split (str): Dataset split to load ("train", "validation", "test")
        cache_dir (str, optional): Custom cache directory path

    Returns:
        datasets.Dataset: Loaded dataset if successful
    """
    try:
        # First attempt: Direct loading
        return datasets.load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    except ValueError as e:
        if "Invalid pattern: '**'" in str(e):
            # Workaround for the '**' glob pattern issue
            if cache_dir is None:
                cache_dir = datasets.config.HF_DATASETS_CACHE

            # Create cache directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)

            # Try loading with explicit download mode
            return datasets.load_dataset(
                dataset_name,
                split=split,
                cache_dir=cache_dir,
                download_mode="force_redownload",
            )
    except Exception as e:
        # Handle other potential errors
        print(f"Error loading dataset {dataset_name}:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")

        # Suggest potential fixes
        if "connection" in str(e).lower():
            print(
                "\nPossible network connectivity issue. Check your internet connection."
            )
        if "permission" in str(e).lower():
            print("\nPossible permissions issue. Check cache directory permissions.")

        raise


# Example usage
try:
    dataset = debug_dataset_loading("stanfordnlp/SHP")
except Exception as e:
    # Handle any remaining errors after attempted fixes
    print("Could not load dataset even with workarounds")
    raise
