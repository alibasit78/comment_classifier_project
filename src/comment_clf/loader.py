import os

import joblib
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer,
    DataCollatorWithPadding,
)

from src.comment_clf.constant import CONFIG_PATH
from src.comment_clf.custom_dataset import CustomDataset

# from constant import CONFIG_PATH
# from custom_dataset import CustomDataset


def create_dataloader(dataset, tokenizer, config, mode):
    """Return the dataloader for the given dataset"""
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    if mode == "train":
        training_set = CustomDataset(dataset, tokenizer, config.train.max_len)
        train_params = {"batch_size": config.train.train_bs, "shuffle": True, "num_workers": 0}
        return DataLoader(training_set, **train_params, collate_fn=data_collator)
        # return DataLoader(training_set, **train_params)
    else:
        testing_set = CustomDataset(dataset, tokenizer, config.train.max_len)
        test_params = {"batch_size": config.train.test_bs, "shuffle": False, "num_workers": 0}
        return DataLoader(testing_set, **test_params, collate_fn=data_collator)
        # return DataLoader(testing_set, **test_params)


if __name__ == "__main__":
    config = OmegaConf.load(CONFIG_PATH)
    train_ds_path = os.path.join(config.data_base_dir, config.data.save_dir, "train.pkl")
    train_size = config.train.train_size

    tokenizer = BertTokenizer.from_pretrained(config.train.tokenizer_name)
    # new_df = pd.read_csv(train_ds_path)
    new_df = joblib.load(train_ds_path)
    train_dataset = new_df.sample(frac=train_size, random_state=config.train.seed)
    test_dataset = new_df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)
    # print(train_dataset.head())
    # print(train_dataset.loc[0, "labels"])
    # print(train_dataset.info())
    train_dataloader = create_dataloader(
        train_dataset, tokenizer=tokenizer, config=config, mode="train"
    )
    print(train_dataloader)
    for data in train_dataloader:
        # print(data)
        print(
            data["input_ids"].shape,
            data["attention_mask"].shape,
            data["token_type_ids"].shape,
            data["targets"].shape,
        )
        break
