import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as data_utils
from tqdm import tqdm
import numpy as np


class CustomDataset(Dataset):

    def __init__(self, dataframe):
        # self.dataframe = dataframe
        self.len = len(dataframe)
        feature = dataframe['feature']
        target = dataframe['target']
        self.X = np.array(feature)
        self.y = np.array(target)

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]).to(torch.float32), torch.tensor(
            self.y[index])

    def __len__(self):
        return self.len


model_path = "./model/facemeshANN.pt"


class ANNClassifier(nn.Module):

    def __init__(self, input_size, output_size, dropout):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_size),
        )

        self.device = ("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using {self.device} device")

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return model, loss


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss, correct


def trainmodel(model,
               train_df,
               val_df,
               test_df,
               epochs=10,
               learning_rate=1e-3,
               batch_size=8):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_dataset = CustomDataset(dataframe=train_df)
    val_dataset = CustomDataset(dataframe=val_df)
    test_dataset = CustomDataset(dataframe=test_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    train_loss = 1
    val_loss = 1
    val_acc = 1
    test_loss = 1
    test_acc = 1
    pbar = tqdm(total=epochs)

    for i in range(epochs):
        pbar.set_description(
            f'Training | train_loss:{train_loss:.4f} | val_loss:{val_loss:.4f} | val_acc:{val_acc:.4f} | test_loss:{test_loss:.4f} | test_acc:{test_acc:.4f}'
        )
        model, train_loss = train_loop(train_loader, model, loss_fn, optimizer)
        if epochs > 10:
            if i % int(0.1 * epochs) == 0:
                test_loss, test_acc = test_loop(test_loader, model, loss_fn)
                
            else:
                val_loss, val_acc = test_loop(val_loader, model, loss_fn)

        pbar.update(1)
    test_loss, test_acc = test_loop(test_loader, model, loss_fn)
    print(
        f'train_loss:{train_loss:.4f} | val_loss:{val_loss:.4f} | val_acc:{val_acc:.4f} | test_loss:{test_loss:.4f} | test_acc:{test_acc:.4f}'
    )
    print("Done!")
    return model, test_loss, test_acc


def testmodel(model, test_df, batch_size=1):
    loss_fn = nn.CrossEntropyLoss()

    test_dataset = CustomDataset(dataframe=test_df)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return test_loop(test_loader, model, loss_fn)


def savemodel(model, save_path='./model.pt'):
    torch.save(model, save_path)
    print("saving done!")


def getmodel(path_to_model):
    model = torch.load(path_to_model)
    model.eval()
    return model