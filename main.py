import os

import joblib
import mlflow
import torch
from mlflow.models import infer_signature
from omegaconf import OmegaConf
from torch import cuda
from transformers import BertTokenizer

from src.comment_clf import logger
from src.comment_clf.constant import CONFIG_PATH
from src.comment_clf.evaluation import compute_eval_report
from src.comment_clf.loader import create_dataloader
from src.comment_clf.model import (
    BERTMODEL,
    freeze_paramater,
    save_model,
)
from src.comment_clf.training import (
    training,
    validation,
)

if __name__ == "__main__":
    config = OmegaConf.load(CONFIG_PATH)
    print("loading config!!")
    train_ds_path = os.path.join(config.data_base_dir, config.data.save_dir, "train.pkl")
    test_ds_path = os.path.join(config.data_base_dir, config.data.save_dir, "test.pkl")
    train_size = config.train.train_size
    tokenizer = BertTokenizer.from_pretrained(config.train.tokenizer_name)
    new_df = joblib.load(train_ds_path)
    test_df = joblib.load(test_ds_path)
    logger.info("train test data loaded!!")
    print("data loaded!!!")
    # new_df = new_df.loc[:64, :]
    print("df: ", new_df.head(), new_df.shape)
    train_dataset = new_df.sample(frac=train_size, random_state=config.train.seed)
    valid_dataset = new_df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)
    test_dataset = test_df  # .loc[:64, :]
    train_dataloader = create_dataloader(
        train_dataset, tokenizer=tokenizer, config=config, mode="train"
    )
    valid_dataloader = create_dataloader(valid_dataset, tokenizer, config, mode="test")
    test_dataloader = create_dataloader(test_dataset, tokenizer, config, mode="test")

    DEVICE = "cuda" if cuda.is_available() else "cpu"
    exp = mlflow.set_experiment(experiment_name="experment_1")
    signature = infer_signature(train_dataset["comment_text"], train_dataset["labels"])
    with mlflow.start_run(experiment_id=exp.experiment_id):
        model = BERTMODEL(config)
        logger.info("bert model is loaded!!")
        print("bert model is loaded!!")
        for param in config.train:
            # print("param: ", param)
            mlflow.log_param(param, config.train[param])
        # model = train(config, model, train_dataloader, DEVICE)
        model, train_loss, valid_loss = training(
            config, model, train_dataloader, valid_dataloader, DEVICE
        )
        # logger.info("training is completed!!!")
        print("training is completed!!!")
        outputs, targets, v_loss = validation(model, test_dataloader, DEVICE)
        # print("outputs: ", outputs)
        # print("targets: ", targets)
        accuracy, f1_score_micro, f1_score_macro = compute_eval_report(
            outputs, targets, message="test"
        )
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("F1_Score_Micro", f1_score_micro)
        mlflow.log_metric("F1_Score_Macro", f1_score_macro)
        mlflow.pytorch.log_model(model, "model", signature=signature)
        save_model(model, os.path.join(config.data_base_dir, config.train.save_dir))
        # mlflow.transformers.log_model(model, "model")
        # components = {
        #     "model": model,
        #     "tokenizer": tokenizer,
        # }
        # mlflow.transformers.log_model(
        #     transformers_model=components,
        #     artifact_path="my_model",
        # )
