import torch
from torch import nn
from torch.utils.data import DataLoader

from preprocessing import load_dataset
from train import train_model, num_trainable_params, evaluate_model

SHALLOW_MODEL_FN  = "baseline_shallow_mlp.pth"
SHALLOW_FIG_FN    = "baseline_shallow_mlp_curves.pdf"
SALLOW_FIG_CONF_FN = "baseline_shallow_mlp_conf.pdf"


class BaseLineMLP(nn.Module):
    def __init__(self, input_size: int, hidden=[]):
        # input_size is is number of pixels in image after flattening
        # hidden is hidden layer sizes (e.g. [100, 100, 50])
        super(BaseLineMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.input_size = input_size
        self.out_size = 4
        hidden = [self.input_size] + hidden + [self.out_size]
        self.layers = self.make_seq(hidden, None)
        print("Baseline MLP model:", self.layers)
        print("Number of trainable parameters:", num_trainable_params(self))
    
    @staticmethod
    def make_seq(dims: list, output_activation: nn.Module) -> nn.Module:
        """Creates a sequential network using ReLUs between layers and no activation at the end

        :param dims (Iterable[int]): tuple in the form of (IN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE2,
            ..., OUT_SIZE) for dimensionalities of layers
        :param output_activation (nn.Module): PyTorch activation function to use after last layer
        :return (nn.Module): return created sequential layers
        """
        mods = []

        for i in range(len(dims) - 2):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            mods.append(nn.ReLU())

        mods.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation:
            mods.append(output_activation())
        return nn.Sequential(*mods)

    def forward(self, x):
        # don't need RGB channels, since they are all duplicates of grayscale channel
        x = x[:, 0, :, :] 
        x = self.flatten(x)
        logits = self.layers(x)
        return logits


if __name__ == "__main__":
    train, valid, test = load_dataset()
    input_size = train[0][0].size()
    input_size = input_size[-1] * input_size[-2] # Width * Height
    shallow_mlp = BaseLineMLP(input_size)

    train_model(train, valid, shallow_mlp, SHALLOW_MODEL_FN , SHALLOW_FIG_FN, max_epochs=100, patience=10)

    shallow_mlp.load_state_dict(torch.load(SHALLOW_MODEL_FN))  # load the best model from train_model
    evaluate_model(test, shallow_mlp, SALLOW_FIG_CONF_FN),   


    




