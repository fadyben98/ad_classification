import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn import svm

from preprocessing import load_dataset, DEVICE, label_str
from train import train_model, num_trainable_params, evaluate_model
from resnet import MODEL_FN as FEATURE_EXTRACTOR_FN

MODEL_FIG_CONF_FN    = "svm_conf.pdf"

class SVM_Wrapper(nn.Module):
    def __init__(self, train: Dataset, valid: Dataset, model_conv: nn.Module):
        super(SVM_Wrapper, self).__init__()
        for param in model_conv.parameters():
            param.requires_grad = False
        
        model_conv.to(DEVICE)
        model_conv.eval()
        clf = svm.SVC(decision_function_shape='ovo')
        X = []
        y = []
        for imgs, labels in DataLoader(train, batch_size=64):
            X.append(model_conv(imgs.to(DEVICE))) # (1, 4)
            y.append(labels)
        
        for imgs, labels in DataLoader(valid, batch_size=64):
            X.append(model_conv(imgs.to(DEVICE))) # (1, 4)
            y.append(labels)

        X = torch.cat(X).cpu()  # (num_samples, 4)
        y = torch.cat(y).cpu()  # (num_samples, )
        clf.fit(X, y)

        clf.decision_function_shape = "ovr"
        self.model_conv = model_conv
        self.clf = clf

    def eval(self):
        self.model_conv.eval()
    
    def forward(self, x: torch.Tensor):
        # x should be a batch of images with shape (batch_size, 3, H, W)
        x = self.model_conv(x)  # x shape => (batch_size, 4)
        x = x.cpu()
        x = self.clf.decision_function(x)
        x = torch.from_numpy(x).to(DEVICE)
        return x


if __name__ == "__main__":
    train, valid, test = load_dataset()
    input_size = train[0][0].size()
    
    model_conv = models.resnet18(pretrained=False)
    num_ftrs = model_conv.fc.in_features
    # replace last fc layer for 4 way classification
    model_conv.fc = nn.Linear(num_ftrs, len(label_str))
    model_conv.load_state_dict(torch.load(FEATURE_EXTRACTOR_FN))  # load feature extractor
    
    svm_model = SVM_Wrapper(train, valid, model_conv)
    evaluate_model(test, svm_model, MODEL_FIG_CONF_FN)
    
    





