import torch
import torchinfo
from omegaconf import OmegaConf
from transformers import BertModel

from src.comment_clf.constant import CONFIG_PATH


def count_parameters(model):
    # table = PrettyTable([“Modules”, “Parameters”])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            print("non trainable paramater: ", name)
            continue
        print("trainable paramater: ", name)
        params = parameter.numel()
        # table.add_row([name, params])
        total_params += params
    # print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def freeze_paramater(model):
    # Freeze the embedding layer and the first 5 transformer layers
    modules_to_freeze = [model.bert.embeddings, *model.bert.encoder.layer[:12]]
    for module in modules_to_freeze:
        for param in module.parameters():
            param.requires_grad = False


def save_model(model, model_path):
    torch.save(model, model_path)


def load_model(model_path):
    return torch.load(model_path)


class BERTMODEL(torch.nn.Module):
    def __init__(self, config):
        super(BERTMODEL, self).__init__()
        self.bert = BertModel.from_pretrained(config.train.model_name)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 6)

    def forward(self, ids, mask, token_type_ids):
        """forward pass"""
        _, output = self.bert(
            ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False
        )
        # print("modl forward: ", output_1.shape)
        output = self.dropout(output)
        output = self.linear(output)
        return output


if __name__ == "__main__":
    config = OmegaConf.load(CONFIG_PATH)
    model = BERTMODEL(config)
    print(model)
    torchinfo.summary(model)
    # for param in model.parameters():
    #     # print(param, )
    #     print(param.)
    freeze_paramater(model)
    count_parameters(model)
