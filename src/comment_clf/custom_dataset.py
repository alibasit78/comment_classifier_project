import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, true_labels=True):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.comment_text
        self.targets = dataframe.labels
        self.max_len = max_len
        self.true_labels = true_labels

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding=True,
            return_token_type_ids=True,
            truncation=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        if self.true_labels:
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                "targets": torch.tensor(self.targets[index], dtype=torch.float),
            }
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        }


# if __name__ == "__main__":
#     CONFIG_PATH = "/home/alibasit/mlops/comment_project/config/config.yaml"
#     config = OmegaConf.load(CONFIG_PATH)
