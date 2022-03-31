import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from preprocessing import load_dataset, DEVICE, label_str
from train import train_model, num_trainable_params, evaluate_model

MODEL_FN          = "transformer.pth"
MODEL_FIG_FN      = "transformer_curves.pdf"
MODEL_FIG_CONF_FN = "transformer_conf.pdf"


if __name__ == "__main__":
    train, valid, test = load_dataset(resize=(224, 224))
    input_size = train[0][0].size()
    
    model_ft = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
    print(model_ft)
    num_ftrs = model_ft.head.in_features
    # replace last fc layer for 4 way classification
    model_ft.head = nn.Linear(num_ftrs, len(label_str))
    print("Number of trainable parameters:", num_trainable_params(model_ft))
    model_ft = model_ft.to(DEVICE)

    train_model(train, valid, model_ft, MODEL_FN , MODEL_FIG_FN, 
        max_epochs=10000, patience=50, batch_size=16, learning_rate=1e-4)

    model_ft.load_state_dict(torch.load(MODEL_FN))  # load the best model from train_model
    evaluate_model(test, model_ft, MODEL_FIG_CONF_FN)
    
    





