# ----------------------- DOWNLOAD FINE-TUNING DATASET ----------------------- #
import pandas as pd
import os
from huggingface_hub import snapshot_download


def ft_download_data(data_name: str, pwd: str = None):
    """
    Download the data from the HuggingFace Hub.

    Args:
        data_name (str): The name of the data to download.
        pwd (str): The current working directory. Defaults to None.
    """
    # Find PWD
    if pwd is None:
        pwd = ".."

    # Determine the path
    if data_name == "sentence_simplification":
        path = os.path.join(pwd, "data", "raw")
        if not os.path.exists(path):
            os.makedirs(path)
    elif data_name == "Data":
        path = os.path.join(pwd, "data", "raw")
        if not os.path.exists(path):
            os.makedirs(path)
    else:
        raise ValueError(f"The data {data_name} is not available.")

    # Download CSVs
    snapshot_download(
        repo_id="OloriBern/FLDE",
        allow_patterns=[f"{data_name}/*.csv"],
        local_dir=path,
        revision="main",
        repo_type="dataset",
    )

    # Return csv paths (recursively)
    csv_paths = [
        os.path.join(path, data_name, file)
        for file in os.listdir(os.path.join(path, data_name))
        if file.endswith(".csv")
    ]
    return csv_paths


def download_data(pwd: str = None):
    csv_path = ft_download_data("sentence_simplification", pwd)
    data = pd.read_csv(
        csv_path[0],
        sep=" -> ",
        names=["Original sentence", "Simplified sentence"],
        header=None,
    )

    return data


# ------------ KEEP ONLY 5 SENTENCE OF EACH LEVEL OF EACH DATASET ------------ #
import pandas as pd


def get_balanced_dataframe(csv_dict: dict, nbr: int = 5):
    if nbr is None:
        nbr = float("inf")

    # Estimate number of sentences
    df_grouped_by_difficulty = [
        df.groupby("Difficulty")
        for key, df in csv_dict.items()
        if "ljl" not in key and "test" not in key
    ]
    to_sample = min(
        nbr, min([int(df.count().min().iloc[0]) for df in df_grouped_by_difficulty])
    )

    # Concatenate all dataframes
    result = pd.concat(
        [df_grouped.sample(to_sample) for df_grouped in df_grouped_by_difficulty]
    )
    # Remove A1
    result = result[result["Difficulty"] != "A1"]
    return result.sort_values(by=["Difficulty"]).reset_index(drop=True)

def download_difficulty_estimation(pwd: str = None):
    csv_path = ft_download_data("Data", pwd)
    csv_dict = {}
    for path in csv_path:
        csv_dict[path.split("/")[-1].split(".")[0]] = pd.read_csv(
            path,
            sep=",",
            names=["Sentence", "Difficulty"],
            header=None,
        ).iloc[1:]

    return csv_dict