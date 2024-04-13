import os
import re
from abc import ABC

import joblib
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.comment_clf.constant import CONFIG_PATH

PUNCTUATION = r'"#$%\(\)\*\+\,/<=>@\[\\\\\]\^_`\{\|\}~\n\:'  # string.punctuation
PUNCTUATION_PATTERN = re.compile("[" + PUNCTUATION + "]")
REPEATED_PUNCTUATION_PATTERN = re.compile(r"\s[!\.\?\'\;\,]+\s")


def clean_text(text):
    """remove the unnessary punctuation from the text"""
    text = PUNCTUATION_PATTERN.sub(" ", text)
    text = REPEATED_PUNCTUATION_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    # text = " ".join(re.findall('['+string.punctuation+"]|\w+", text))
    return text.strip()


class RawDataset(ABC):
    def load_data(self):
        """load data from the df"""
        pass


class TrainTestRawDataset(RawDataset):
    def __init__(self, config):
        self.config = config

    def load_data_util(self, df):
        """common utility function for both train and test df"""
        column_names = df.columns[2:]
        df["labels"] = df[df.columns[2:]].values.tolist()
        new_df = df[["comment_text", "labels"]].copy()
        return new_df, column_names

    def load_data(self, mode="train"):
        if mode == "train":
            df = pd.read_csv(os.path.join(self.config.data_base_dir, self.config.data.train_data))
            new_df, column_names = self.load_data_util(df)
            return new_df, column_names
        if mode == "test":
            df = pd.read_csv(os.path.join(self.config.data_base_dir, self.config.data.test_data))
            df_labels = pd.read_csv(
                os.path.join(self.config.data_base_dir, self.config.data.test_labels)
            )
            new_df = pd.merge(left=df, right=df_labels, how="inner", on="id")
            new_df, column_names = self.load_data_util(new_df)
            return new_df, column_names
        return None, None

    def save_data(self, df, filename):
        path = os.path.join(self.config.data_base_dir, self.config.data.save_dir)
        os.makedirs(path, exist_ok=True)
        # df.to_csv(
        #     os.path.join(path, filename),
        #     index=False,
        # )
        joblib.dump(df, os.path.join(path, filename))


def pre_process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the column_text column of the dataframe and drop the duplicates row
    """
    # print("pre_process")
    df.loc[:, "comment_text"] = df["comment_text"].apply(clean_text)
    df.drop_duplicates(subset="comment_text", keep="first", inplace=True)
    df = df[~(df["comment_text"] == "")]
    df["labels"] = df["labels"].apply(lambda x: list(np.where(np.array(x) == -1, 0, np.array(x))))
    return df


if __name__ == "__main__":
    # CONFIG_PATH = "/home/alibasit/mlops/comment_project/config/config.yaml"
    config = OmegaConf.load(CONFIG_PATH)
    raw_dataset = RawDataset()
    raw_dataset = TrainTestRawDataset(config=config)
    train_df, label_names = raw_dataset.load_data(mode="train")
    test_df, _ = raw_dataset.load_data(mode="test")
    print(train_df.head())
    print(label_names)
    print(test_df.head())
    # train_df = pre_process_data(train_df)
    train_df = pre_process_data(train_df)
    test_df = pre_process_data(test_df)
    print("train_df: ", train_df.head())
    print("test_df: ", test_df.head())
    print("shape of train: ", train_df.shape)
    print("shape of test: ", test_df.shape)
    raw_dataset.save_data(train_df, "train.pkl")
    raw_dataset.save_data(test_df, "test.pkl")
