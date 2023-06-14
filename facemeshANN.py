import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as data_utils
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt


class CustomDataset(Dataset):

    def __init__(self, dataframe):
        # self.dataframe = dataframe
        self.len = len(dataframe)
        feature = dataframe['feature']
        target = dataframe['target']
        self.device = ("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
        self.X = torch.from_numpy(np.array([f for f in feature])).to(torch.float32).to(self.device)
        self.y = torch.tensor([torch.tensor(t) for t in target]).to(self.device)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len

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
            nn.Linear(512, 512),
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
        x = self.linear_relu_stack(x)
        return x


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    training_loss = 0
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
        
    return model, training_loss/len(dataloader)


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
               lr=1e-4,
               batch_size=8):
    device = model.device
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=int(epochs/2))

    train_dataset = CustomDataset(dataframe=train_df)
    val_dataset = CustomDataset(dataframe=val_df)
    test_dataset = CustomDataset(dataframe=test_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True)
    
    train_loss = torch.tensor(0)
    val_loss = torch.tensor(0)
    val_acc = torch.tensor(0)
    test_loss = torch.tensor(0)
    test_acc = torch.tensor(0)
    train_loss_backup = []
    val_loss_backup = []
    val_acc_backup = []
    test_loss_backup = []
    test_acc_backup = []
    
    pbar = tqdm(total=epochs)

    model.to(device)

    for i in range(epochs):
        pbar.set_description(
            f'Epoch {i+1} | tr_loss:{train_loss:.4f} | va_loss:{val_loss:.4f} | va_acc:{val_acc:.4f} | te_loss:{test_loss:.4f} | te_acc:{test_acc:.4f}'
        )
        model, train_loss = train_loop(train_loader, model, loss_fn, optimizer)
        if i == 0 :
            val_loss, val_acc = test_loop(val_loader, model, loss_fn)
            test_loss, test_acc = test_loop(test_loader, model, loss_fn)
            
        if epochs > 10:
            if i % 10 == 0:
                test_loss, test_acc = test_loop(test_loader, model, loss_fn)
            else:
                val_loss, val_acc = test_loop(val_loader, model, loss_fn)
                
        scheduler.step()
        train_loss_backup.append(train_loss)
        test_loss_backup.append(test_loss)
        test_acc_backup.append(test_acc)
        val_loss_backup.append(val_loss)
        val_acc_backup.append(val_acc)
        pbar.update(1)
              
        
    test_loss, test_acc = test_loop(test_loader, model, loss_fn)
    print(
        f'Result : train_loss:{train_loss:.4f} | val_loss:{val_loss:.4f} | val_acc:{val_acc:.4f} | test_loss:{test_loss:.4f} | test_acc:{test_acc:.4f}'
    )
    print("Done!")
    # plt.subplot(1,2,1)
    # plt.plot(range(epochs), val_loss_backup, label = "val_loss")
    # plt.plot(range(epochs), test_loss_backup, label = "test_loss")
    # plt.plot(range(epochs), train_loss_backup, label = "train_loss")
    # plt.legend()
    # plt.subplot(1,2,2)
    # plt.plot(range(epochs), val_acc_backup, label = "val_acc")
    # plt.plot(range(epochs), test_acc_backup, label = "test_acc")
    # plt.legend()
    # plt.show()
    
    model.eval()
    
    return model, test_loss, test_acc


def testmodel(model, test_df, batch_size=1):
    loss_fn = nn.CrossEntropyLoss()
    
    test_dataset = CustomDataset(dataframe=test_df)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return test_loop(test_loader, model, loss_fn)


def savemodel(model, save_path='./model.pt'):
    torch.save(model.state_dict(), save_path)
    print("saving done!")


def getmodel(model, path_to_model):
    model.load_state_dict(torch.load(path_to_model))
    model.device = ("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(model.device)
    model.eval()
    return model

def predict(model, X):
    return model(torch.from_numpy(X).to(torch.float32).to(model.device))