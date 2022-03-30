import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision import transforms, datasets
from torch.utils.data import random_split
from typing import Tuple

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")

DATASET_DIR = "dataset"
TRAIN_DIR = DATASET_DIR + os.sep + "train"
TEST_DIR = DATASET_DIR + os.sep + "test"

# idx is label number
label_str = ["Mild", "Moderate", "Normal", "Very Mild"]

def get_counts(dataset: datasets.ImageFolder, save_fn="grid_of_images.pdf"):
    # get counts and display grid of images with labels
    counts = {}
    for _, label in dataset:
        st = label_str[label]
        if st not in counts:
            counts[st] = 0
        counts[st] += 1
    print(counts)

    if save_fn is None:
        return  # don't display images

    # display 3 images for each label
    grid_size = (4, 3)
    bounderies = [0, counts[label_str[0]], counts[label_str[0]] + counts[label_str[1]], 
                  counts[label_str[0]] + counts[label_str[1]] + counts[label_str[2]]]
    imgs = [[dataset[idx + i] for i in range(grid_size[1])] for idx in bounderies] 
    
    fig = plt.figure(constrained_layout=True)
    fig.suptitle('3 MRI images from each class')

    subfigs = fig.subfigures(nrows=grid_size[0], ncols=1)
    for row, subfig in enumerate(subfigs):
        label = label_str[imgs[row][0][1]]
        subfig.suptitle(label)

        axs = subfig.subplots(nrows=1, ncols=grid_size[1])
        for col, ax in enumerate(axs):
            im = imgs[row][col][0][0].numpy()
            ax.imshow(im)
    plt.savefig(save_fn)
    plt.show()


def load_dataset(train_split=0.8, display_images=False) -> Tuple[datasets.ImageFolder]:
    preprocess_transforms = transforms.Compose([transforms.ToTensor()])

    # train and test will have 3 channels (3, H, W), but since the image is grayscale
    # the 3 channels have exactly the same values

    train = datasets.ImageFolder("dataset/train", transform=preprocess_transforms)

    if 0.0 < train_split < 1.0: # split into training and validation sets
        num_train_samples = int(train_split * len(train))
        num_val_samples = len(train) - num_train_samples
        train, valid = random_split(train, [num_train_samples, num_val_samples], 
                                    generator=torch.Generator().manual_seed(42))
    else:
        valid = None

    test = datasets.ImageFolder("dataset/test", transform=preprocess_transforms)

    if display_images:
        get_counts(train)
        if valid is not None:
            get_counts(valid, None)
        get_counts(test, "grid_of_images_test.pdf")

    return train, valid, test


    
if __name__ == "__main__":
    load_dataset(display_images=True)
    
    
    


