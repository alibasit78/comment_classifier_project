import os

import joblib
import torch
from omegaconf import OmegaConf
from torch import cuda
from transformers import BertTokenizer

from src.comment_clf import logger
from src.comment_clf.constant import CONFIG_PATH
from src.comment_clf.evaluation import compute_eval_report
from src.comment_clf.loader import create_dataloader
from src.comment_clf.model import (
    BERTMODEL,
    count_parameters,
    freeze_paramater,
)


def loss_fn(outputs, targets):
    """multi label binary loss function"""
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def validation(model, testing_loader, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    valid_loss = 0.0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data["input_ids"].to(device, dtype=torch.long)
            mask = data["attention_mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            loss = loss_fn(outputs, targets)
            valid_loss += loss.item()
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets, valid_loss / len(testing_loader)


def train(config, epoch, model, training_loader, device):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.train.learning_rate)
    model.train()
    for i, data in enumerate(training_loader, 0):
        ids = data["input_ids"].to(device, dtype=torch.long)
        mask = data["attention_mask"].to(device, dtype=torch.long)
        token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
        targets = data["targets"].to(device, dtype=torch.float)

        print(f"ids: {ids.shape}")

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if i % 10 == 0:
            # print(f"Epoch: {epoch}, Loss:  {loss.item()}")
            print("steps: ", i)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, loss.item() / len(training_loader)


def training(config, model, training_loader, test_dataloader, device):
    """training the bert model"""
    freeze_paramater(model)
    count_parameters(model)
    train_loss = []
    valid_loss = []
    for epoch in range(config.train.epochs):
        model, loss = train(config, epoch, model, training_loader, device)
        train_loss.append(loss)
        print("EPOCH: ", epoch, "loss: ", loss)
        outputs, targets, v_loss = validation(model, test_dataloader, device)
        valid_loss.append(v_loss)
        compute_eval_report(outputs, targets, message="validation")
    return model, train_loss, valid_loss


if __name__ == "__main__":
    config = OmegaConf.load(CONFIG_PATH)
    train_ds_path = os.path.join(config.data_base_dir, config.data.save_dir, "train.pkl")
    train_size = config.train.train_size

    tokenizer = tokenizer = BertTokenizer.from_pretrained(config.train.tokenizer_name)
    new_df = joblib.load(train_ds_path)
    logger.info("train test data loaded!!")
    new_df = new_df.loc[:16, :]
    print("df: ", new_df.head(), new_df.shape)
    train_dataset = new_df.sample(frac=train_size, random_state=config.train.seed)
    test_dataset = new_df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    train_dataloader = create_dataloader(
        train_dataset, tokenizer=tokenizer, config=config, mode="train"
    )
    test_dataloader = create_dataloader(test_dataset, tokenizer, config, mode="test")

    DEVICE = "cuda" if cuda.is_available() else "cpu"
    model = BERTMODEL(config)
    logger.info("bert model is loaded!!")
    print("bert model is loaded!!")
    model = training(config, model, train_dataloader, DEVICE)
    # logger.info("training is completed!!!")
    print("training is completed!!!")
    outputs, targets = validation(model, test_dataloader, DEVICE)
    compute_eval_report(outputs, targets)
