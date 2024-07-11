import os

import joblib
import mlflow
import torch

# from mlflow.models import infer_signature
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
from src.comment_clf.training import (
    training,
    validation,
)

if __name__ == "__main__":
    # logger.setLevel(logging.INFO)
    config = OmegaConf.load(CONFIG_PATH)
    # print("loading config!!", logger)
    logger.info("loading config")
    train_ds_path = os.path.join(config.data_base_dir, config.data.save_dir, "train.pkl")
    test_ds_path = os.path.join(config.data_base_dir, config.data.save_dir, "test.pkl")
    train_size = config.train.train_size
    tokenizer = BertTokenizer.from_pretrained(config.train.tokenizer_name)
    new_df = joblib.load(train_ds_path)
    test_df = joblib.load(test_ds_path)
    logger.info("train test data loaded!!")
    # print("data loaded!!!")
    # new_df = new_df.loc[:64, :]
    print("df: ", new_df.head(), new_df.shape)

    train_dataset = new_df.sample(frac=train_size, random_state=config.train.seed)
    valid_dataset = new_df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    train_dataset = train_dataset.loc[:64, :]
    test_dataset = test_df.loc[:32, :]
    valid_dataset = valid_dataset.loc[:32, :]

    train_dataloader = create_dataloader(
        train_dataset, tokenizer=tokenizer, config=config, mode="train"
    )
    valid_dataloader = create_dataloader(valid_dataset, tokenizer, config, mode="test")
    test_dataloader = create_dataloader(test_dataset, tokenizer, config, mode="test")

    DEVICE = torch.device("cuda") if cuda.is_available() else torch.device("cpu")
    exp = mlflow.set_experiment(experiment_name="experment_1")
    # signature = infer_signature(train_dataset["comment_text"], train_dataset["labels"])
    if config.train.training:
        with mlflow.start_run(experiment_id=exp.experiment_id):
            model = BERTMODEL(config)
            freeze_paramater(model)
            count_parameters(model)
            logger.info("bert model is loaded!!")
            # print("bert model is loaded!!")
            params = {param: config.train[param] for param in config.train}
            # print("param: ", param)
            mlflow.log_params(params)
            # model = train(config, model, train_dataloader, DEVICE)
            model, train_loss, valid_loss = training(
                config, model, train_dataloader, valid_dataloader, DEVICE, mlflow
            )
            logger.info("training is completed!!!")
            # print("training is completed!!!")
            outputs, targets, v_loss = validation(model, test_dataloader, DEVICE)
            # print("outputs: ", outputs)
            # print("targets: ", targets)
            accuracy, f1_score_micro, f1_score_macro = compute_eval_report(
                outputs, targets, message="val"
            )
            metrics = {
                "accuracy": accuracy,
                "f1_score_micro": f1_score_micro,
                "f1_score_macro": f1_score_macro,
            }
            mlflow.log_metrics(metrics)
            current_run = mlflow.active_run()
            logger.info(f"active run id is: {current_run.info.run_id}")
            logger.info(f"active run id is: {current_run.info.run_name}")
    else:
        with mlflow.start_run(experiment_id=exp.experiment_id):
            logger.info("loading the trained model!")
            model = mlflow.pytorch.load_model(
                model_uri="runs:/f8bb0cda3a4c42c2b752395c0596de85/pytorch_model"
            )
            logger.info("testing!!")
            outputs, targets, v_loss = validation(model, test_dataloader, DEVICE)
            accuracy, f1_score_micro, f1_score_macro = compute_eval_report(
                outputs, targets, message="test"
            )
