import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from preprocessing import load_dataset, DEVICE
from train import train_model, num_trainable_params, evaluate_model

MODEL_FN          = "cnn_shallow_mlp.pth"
MODEL_FIG_FN      = "cnn_shallow_mlp_curves.pdf"
MODEL_FIG_CONF_FN = "cnn_shallow_mlp_conf.pdf"



class PadMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride, return_indices=False, return_pad=False):
        super(PadMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool = nn.MaxPool2d(kernel_size, stride, return_indices=return_indices)
        self.pad = nn.ConstantPad2d(padding=0, value=0)
        self.return_indices = return_indices
        self.return_pad = return_pad

    def set_new_return(self, return_indices=True, return_pad=True):
        self.return_indices = return_indices
        self.return_pad = return_pad
        self.pool.return_indices = return_indices

    def forward(self, f_maps):
        coords = [self.stride - f_maps.size(i + 2) % self.stride for i in range(2)]
        for i, coord in enumerate(coords):
            if coord == self.stride:
                coords[i] = 0

        self.pad.padding = (coords[1], 0, coords[0], 0)

        if self.return_indices:
            output, indices = self.pool(self.pad(f_maps))

            if self.return_pad:
                return output, indices, (coords[1], 0, coords[0], 0)
            else:
                return output, indices

        else:
            output = self.pool(self.pad(f_maps))

            if self.return_pad:
                return output, (coords[1], 0, coords[0], 0)
            else:
                return output


class Conv5_FC3(nn.Module):
    def __init__(self, input_size, output_size=4, dropout=0.5):
        # input_size is img dimension e.g. (1, 180, 150)
        super(Conv5_FC3, self).__init__()
        conv, norm, pool = nn.Conv2d, nn.BatchNorm2d, PadMaxPool2d
        self.convolutions = nn.Sequential(
            conv(input_size[0], 8, 3, padding=1),
            norm(8),
            nn.ReLU(),
            pool(2, 2),

            conv(8, 16, 3, padding=1),
            norm(16),
            nn.ReLU(),
            pool(2, 2),

            conv(16, 32, 3, padding=1),
            norm(32),
            nn.ReLU(),
            pool(2, 2),

            conv(32, 64, 3, padding=1),
            norm(64),
            nn.ReLU(),
            pool(2, 2),

            conv(64, 128, 3, padding=1),
            norm(128),
            nn.ReLU(),
            pool(2, 2),
        ).to(DEVICE)

        # Compute the size of the first FC layer
        input_tensor = torch.zeros(input_size).unsqueeze(0)
        input_tensor = input_tensor.to(DEVICE)
        output_convolutions = self.convolutions(input_tensor)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),

            nn.Linear(np.prod(list(output_convolutions.shape)).item(), 1300),
            nn.ReLU(),

            nn.Linear(1300, 50),
            nn.ReLU(),

            nn.Linear(50, output_size)
        ).to(DEVICE)
        
        print("CNN model:", self.convolutions, self.fc)
        print("Number of trainable parameters:", num_trainable_params(self))


    def forward(self, x):
        # don't need RGB channels, since they are all duplicates of grayscale channel
        x = x[:, :1, :, :] 
        x = self.convolutions(x)
        logits = self.fc(x)
        return logits


if __name__ == "__main__":
    train, valid, test = load_dataset()
    input_size = train[0][0].size()
    input_size = (1, input_size[1], input_size[2])  # don't want rgb channel
    
    cnn = Conv5_FC3(input_size)

    train_model(train, valid, cnn, MODEL_FN , MODEL_FIG_FN, max_epochs=20000, patience=200)

    cnn.load_state_dict(torch.load(MODEL_FN))  # load the best model from train_model
    evaluate_model(test, cnn, MODEL_FIG_CONF_FN)
    
    





