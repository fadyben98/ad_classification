import torch
import numpy as np
import itertools
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix 

from preprocessing import label_str, DEVICE


def num_trainable_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_confusion_matrix(cm, save_fn: str, classes=label_str, normalize=False, 
                          title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(len(classes), len(classes)))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm, "\n")
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_fn)
    plt.show()


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()  # turn on drop-out etc
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss, total_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        
        # Augment data and send to correct device (cpu or gpu)
        if np.random.rand() < 0.5:  # augment data with 50% chance
            brightness = 0.05 * np.random.randn()
            black_pxl_val = X[0, 0, 0, 0]
            black_idxs = X == black_pxl_val

            X = X + brightness
            X[black_idxs] = black_pxl_val

        X = X.to(DEVICE)
        y = y.to(DEVICE)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            total_loss += loss.item()
            total_acc += (pred.argmax(1) == y).type(torch.float).sum().item()

    total_loss /= num_batches
    total_acc /= size
    return total_loss, total_acc


def test_loop(dataloader, model, loss_fn, get_conf_matrix=False):
    model.eval()  # turn off drop-out etc
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    all_ys, all_preds = [], []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            if get_conf_matrix:
                all_ys.append(y)
                all_preds.append(pred.argmax(1))
        
        report = None
        conf_matrix = None
        if get_conf_matrix:
            all_ys = torch.cat(all_ys).cpu()
            all_preds = torch.cat(all_preds).cpu()
            conf_matrix = confusion_matrix(all_ys, all_preds)
            report = classification_report(all_ys, all_preds, digits=3)

    test_loss /= num_batches
    correct /= size
    return test_loss, correct, conf_matrix, report


def train_model(train: Dataset, valid: Dataset, model: nn.Module, 
                model_fn: str, fig_fn: str,
                learning_rate=1e-3, batch_size=64, max_epochs=1000, patience=20):
    # save_fn is file name to use when saving the model parameters (.pth extension)
    # fig_fn is file name to use when saving training curves (.pdf extension)
    # patience is num epochs allowed without val_loss improvement (used for early stopping)

    train_dataloader = DataLoader(train, batch_size=batch_size)
    valid_dataloader = DataLoader(valid, batch_size=batch_size)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    no_improvement = 0  # num epochs without improvement

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    
    for t in range(max_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss, train_acc = train_loop(train_dataloader, model, loss_fn, optimizer)
        val_loss, val_acc, _, _ = test_loop(valid_dataloader, model, loss_fn)
        print(f"Train: \n Accuracy: {(100*train_acc):>0.1f}%, Avg loss: {train_loss:>8f} \n")
        print(f"Validation: \n Accuracy: {(100*val_acc):>0.1f}%, Avg loss: {val_loss:>8f} \n")
        print(f"Epochs without val_loss improvement:  {no_improvement} \n")

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_loss < best_val_loss:
            no_improvement = 0 
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_fn)
        else:
            no_improvement += 1
        
        if no_improvement > patience:
            print("\nEarly stopping: no improvement on val_loss for", patience, "epochs\n")
            print(f"Best Validation Loss: {best_val_loss:>8f} \n")
            break

    model.eval()  # turn off drop-out etc
    # plot train loss, val loss, train_acc and val_acc side by side
    plt.subplot(1, 2, 1) # row 1, col 2 index 1
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Loss Training Curve")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2) # index 2
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Validation Acc")
    plt.title("Accuracy Training Curve")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()

    plt.savefig(fig_fn)
    plt.show()


def evaluate_model(test: Dataset, model: nn.Module, conf_matrix_fn: str, batch_size=64):
    # evaluate the model by printing test loss, test acc, test F1 and confusion matrix
    # WARNING: only evaluate on test set once per model (use validation set for tuning not test set)
    test_dataloader = DataLoader(test, batch_size=batch_size)
    loss_fn = nn.CrossEntropyLoss()

    test_loss, test_acc, conf_matrix, report = test_loop(test_dataloader, model, loss_fn, get_conf_matrix=True)
    print(f"Test: \n Accuracy: {(100*test_acc):>0.2f}%, Avg loss: {test_loss:>8f} \n")
    print(report)
    plot_confusion_matrix(conf_matrix, save_fn=conf_matrix_fn)







