from __future__ import annotations

import argparse
import os

import datasets


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python scripts/download/download_dolly.py")
    p.add_argument("--dataset_name", type=str, default="databricks/databricks-dolly-15k")
    p.add_argument("--output_dir", type=str, default=None, help="Output directory. Defaults to data/raw_data/<dataset_name_suffix>.")
    p.add_argument("--cache_dir", type=str, default="data/hf_datasets")
    args = p.parse_args(argv)

    dataset_name = args.dataset_name
    output_dir = args.output_dir
    if not output_dir:
        # e.g. "databricks-dolly-15k" from "databricks/databricks-dolly-15k"
        # or simplified to "dolly" if it matches common names
        suffix = dataset_name.split("/")[-1]
        if "dolly" in suffix.lower():
            suffix = "dolly"
        output_dir = os.path.join("data", "raw_data", suffix)

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading dataset '{dataset_name}' to cache '{args.cache_dir}'...")
    # Load dataset
    ds = datasets.load_dataset(dataset_name, cache_dir=args.cache_dir)
    
    # Typically dolly has 'train', let's check splits and save
    if isinstance(ds, datasets.DatasetDict):
        if "train" in ds:
            train_ds = ds["train"]
        else:
            # Fallback to first split
            first_split = list(ds.keys())[0]
            train_ds = ds[first_split]
    else:
        train_ds = ds

    print(f"Saving dataset to disk at '{output_dir}'...")
    train_ds.save_to_disk(output_dir)
    print(f"Done! Saved to: {output_dir}")


if __name__ == "__main__":
    main()
