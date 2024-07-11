import numpy as np
from sklearn import metrics

from src.comment_clf import logger

# from src.comment_clf import logger


def compute_eval_report(outputs, targets, message="test"):
    outputs = np.array(outputs) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average="micro", zero_division=0)
    f1_score_macro = metrics.f1_score(targets, outputs, average="macro", zero_division=0)
    logger.info(f"Accuracy Score = {accuracy}")
    logger.info(f"F1 Score (Micro) = {f1_score_micro}")
    logger.info(f"F1 Score (Macro) = {f1_score_macro}")
    # print(f"Results of {message}")
    # print(f"Accuracy Score = {accuracy}")
    # print(f"F1 Score (Micro) = {f1_score_micro}")
    # print(f"F1 Score (Macro) = {f1_score_macro}")
    return accuracy, f1_score_micro, f1_score_macro
