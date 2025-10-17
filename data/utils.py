from datasets import load_dataset
import pandas as pd
from collections import defaultdict
from datasets import Dataset
from typing import Tuple, Literal
import numpy as np
from datasets import Dataset, Features, Sequence, Value
import math

def load_Amazon_dataset_hf(name, only_rating=True):
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{name}", trust_remote_code=True)["full"]
    if only_rating:
        dataset = dataset.select_columns(["parent_asin", "user_id", "timestamp"])
    return dataset

def filter_K_core(ds: Dataset, K: int = 5) -> Dataset:
    """
    Keep only interactions where BOTH the user and the item (parent_asin)
    appear at least K times. Returns a new Hugging Face Dataset.

    Parameters
    ----------
    ds : datasets.Dataset
        Must contain 'user_id' and 'parent_asin' columns.
    K : int
        Minimum interactions per user and per item.

    Returns
    -------
    datasets.Dataset
        Dataset satisfying the K-core constraint.
    """
    # Convert once to pandas for fast counting
    df = ds.to_pandas()

    # Iteratively prune until stable (true k-core)
    prev_len = -1
    while True:
        user_counts = df["user_id"].value_counts()
        item_counts = df["parent_asin"].value_counts()

        valid_users = user_counts.index[user_counts >= K]
        valid_items = item_counts.index[item_counts >= K]

        new_df = df[df["user_id"].isin(valid_users) & df["parent_asin"].isin(valid_items)]

        if len(new_df) == prev_len:
            df = new_df
            break
        prev_len = len(new_df)
        df = new_df

    # Back to HF Dataset (no extra index column)
    return Dataset.from_pandas(df.reset_index(drop=True), preserve_index=False)

def create_item_id_mappings(ds):
    """Create two dictionaries:
    - asin2id: maps parent_asin -> index
    - id2asin: maps index -> parent_asin
    """
    unique_asins = list(set(ds["parent_asin"]))
    asin2id = {asin: idx for idx, asin in enumerate(unique_asins)}
    id2asin = {idx: asin for asin, idx in asin2id.items()}
    return asin2id, id2asin

def create_plain_sequences_dataset(ds, asin2id):
    """
    Create a new dataset with a single column 'sequences',
    where each sequence corresponds to the ordered list of item IDs
    a user interacted with (based on timestamp).
    """
    user_history = defaultdict(list)

    # Step 1: group items per user
    for sample in ds:
        user_id = sample["user_id"]
        asin = sample["parent_asin"]
        ts = sample["timestamp"]
        user_history[user_id].append((ts, asin))

    # Step 2: sort and convert ASINs to IDs
    sequences = []
    for user_id, items in user_history.items():
        sorted_items = sorted(items, key=lambda x: x[0])  # sort by timestamp
        item_ids = [asin2id[a] for _, a in sorted_items if a in asin2id]
        if len(item_ids) > 0:
            sequences.append({"sequences": item_ids})

    # Step 3: convert to Hugging Face Dataset
    new_ds = Dataset.from_list(sequences)
    return new_ds

def split_train_test(
    ds: Dataset,
    split_type: Literal["user", "leave_one_out"] = "user",
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Split a HF Dataset with a 'sequences' column (list[int]) for sequence recommendation.

    Modes
    -----
    user:
        Non-overlapping temporal split by ratios:
            train = seq[:T], valid = seq[T:V], test = seq[V:]
        Defaults to ~80/10/10 per row.

    leave_one_out:
        Classic LOO for seq length >=3:
            train = seq[:-2], valid = seq[:-1], test = seq[:]
        (rows with len<3 are skipped)

    Returns
    -------
    (train_ds, valid_ds, test_ds) : Hugging Face Datasets
    """
    if "sequences" not in ds.column_names:
        raise ValueError("Expected a 'sequences' column containing list[int].")

    # optionally carry user_id if present
    carry_user = "user_id" in ds.column_names
    features = Features({
        "sequences": ds.features.get("sequences", Sequence(Value("int64"))),
        **({"user_id": ds.features.get("user_id", Value("string"))} if carry_user else {})
    })

    def make_ds(rows):
        if not rows:
            cols = {"sequences": []}
            if carry_user: cols["user_id"] = []
            return Dataset.from_dict(cols, features=features)
        cols = {"sequences": [r["sequences"] for r in rows]}
        if carry_user: cols["user_id"] = [r["user_id"] for r in rows]
        return Dataset.from_dict(cols, features=features)

    train_rows, valid_rows, test_rows = [], [], []

    if split_type == "leave_one_out":
        for ex in ds:
            seq = list(ex["sequences"])
            n = len(seq)
            if n < 3:
                continue
            base = {"user_id": ex["user_id"]} if carry_user else {}
            train_rows.append({"sequences": seq[:-2], **base})
            valid_rows.append({"sequences": seq[:-1], **base})
            test_rows.append({"sequences": seq[:], **base})

    elif split_type == "user":
        tr, va, te = ratios
        if not (tr > 0 and va >= 0 and te > 0 and abs(tr + va + te - 1.0) < 1e-6):
            raise ValueError("ratios must be >0 and sum to 1.0, e.g., (0.8, 0.1, 0.1).")

        import numpy as np, math
        N = len(ds)
        rng = np.random.default_rng(42)  # fixed seed for reproducibility
        perm = np.arange(N)
        rng.shuffle(perm)

        n_train = int(math.floor(N * tr))
        n_valid = int(math.floor(N * va))
        n_test  = N - n_train - n_valid  # ensure all rows assigned

        train_idx = set(perm[:n_train])
        valid_idx = set(perm[n_train:n_train + n_valid])
        test_idx  = set(perm[n_train + n_valid:])

        for i, ex in enumerate(ds):
            seq = list(ex["sequences"])
            base = {"user_id": ex["user_id"]} if carry_user else {}
            row = {"sequences": seq, **base}

            if i in train_idx:
                train_rows.append(row)
            elif i in valid_idx:
                valid_rows.append(row)
            else:
                test_rows.append(row)

    else:
        raise ValueError('split_type must be "user" or "leave_one_out".')

    return make_ds(train_rows), make_ds(valid_rows), make_ds(test_rows)

def create_labels(ds: Dataset) -> Dataset:
    """
    For each sample in the dataset, create:
    - 'seq': all elements except the last
    - 'label': the last element

    Args:
        ds (Dataset): Hugging Face dataset with column 'sequences'
                      (list of integers per example)
    Returns:
        Dataset: new dataset with 'seq' and 'label' columns
    """
    def split_seq(example):
        seq = example["sequences"][:-1]
        label = example["sequences"][-1]
        return {"seq": seq, "label": label}
    
    new_ds = ds.map(split_seq, remove_columns=["sequences"])
    return new_ds


def prepare_data(name, K=5, split_type: Literal["user", "leave_one_out"] = "user",
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
    ds = load_Amazon_dataset_hf(name)
    if K:
        ds = filter_K_core(ds, K)
    asin2id, id2asin = create_item_id_mappings(ds)
    ds = create_plain_sequences_dataset(ds, asin2id)
    ds_train, ds_valid, ds_test = split_train_test(ds, split_type, ratios)
    ds_train = create_labels(ds_train)
    ds_valid = create_labels(ds_valid)
    ds_test = create_labels(ds_test)
    return ds_train, ds_valid, ds_test

    