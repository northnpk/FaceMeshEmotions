import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


class CustomDataset(Dataset):

    def __init__(self, dataframe, device='auto'):
        self.len = len(dataframe)
        feature = dataframe['feature']
        edge_index = dataframe['edge_index'][0]
        target = dataframe['target']
        if device == 'auto':
            self.device = ("cuda" if torch.cuda.is_available() else
                           "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
        X = torch.from_numpy(np.array([f for f in feature])).to(
            torch.float32)
        edge_index = torch.from_numpy(
            np.array(edge_index)).t().contiguous().to(torch.long)
        y = [torch.tensor(t).to(self.device) for t in target]
        self.data = [Data(x=X[i], edge_index=edge_index, y=y[i]).to(self.device) for i in range(self.len)]
        classweight = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(target),
            y=np.array(target))
        print(f'Class weight:{classweight}')
        self.class_weights = torch.tensor(classweight,
                                          dtype=torch.float).to(self.device)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class GCNClassifier(nn.Module):

    def __init__(self, input_size, output_size, dropout, device='auto'):
        super().__init__()
        self.conv1 = GCNConv(input_size, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 32)
        self.lin = nn.Linear(32, output_size)
        self.dropout = dropout

        if device == 'auto':
            self.device = ("cuda" if torch.cuda.is_available() else
                           "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Using {self.device} device")

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return x


def train_loop(dataloader, model, loss_fn, optimizer):
    size = 0
    training_loss, correct = 0, 0
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for data in dataloader:
        # Compute prediction and loss
        pred = model(data.x, data.edge_index, data.batch)
        loss = loss_fn(pred, data.y)
        correct += (pred.argmax(1) == data.y).type(torch.float).sum().item()
        size += len(data.y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss += loss.item()

    return model, training_loss / len(dataloader), correct / size


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = 0
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for data in dataloader:
            pred = model(data.x, data.edge_index, data.batch)
            test_loss += loss_fn(pred, data.y).item()
            correct += (pred.argmax(1) == data.y).type(torch.float).sum().item()
            size += len(data.y)

    test_loss /= num_batches
    correct /= size
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss, correct


def confusematrixtest(dataloader, model, loss_fn, class_name):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = 0
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    pred_list = []
    y_list = []

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for data in dataloader:
            pred = model(data.x, data.edge_index, data.batch)
            test_loss += loss_fn(pred, data.y).item()
            correct += (pred.argmax(1) == data.y).type(torch.float).sum().item()
            size += len(data.y)
            pred_list.extend(pred.argmax(1).cpu())
            y_list.extend(data.y.cpu())

    test_loss /= num_batches
    correct /= size
    # print(f'Preds:{np.array(pred_list)}')
    # print(f'y:{np.array(y_list)}')
    cm = confusion_matrix(np.array(y_list), np.array(pred_list))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_name)
    disp.plot()
    plt.show()
    print(classification_report(y_list, pred_list, target_names=class_name))


def trainmodel(model,
               train_df,
               val_df,
               test_df,
               epochs=10,
               lr=1e-4,
               batch_size=8,
               plot=False,
               class_name=None):
    device = model.device
    print(f'devices:{device}')

    train_dataset = CustomDataset(dataframe=train_df, device=device)
    val_dataset = CustomDataset(dataframe=val_df, device=device)
    test_dataset = CustomDataset(dataframe=test_df, device=device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    class_weights = torch.tensor([
        1.0971897, 19.12244898, 1.08826945, 0.53757889, 0.81125541, 1.26280323,
        0.81125541
    ],
                                 dtype=torch.float).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=10,
        threshold=0.0001,
        threshold_mode='rel',
        cooldown=0,
        min_lr=4e-6,
        eps=1e-08)

    train_loss = torch.tensor(0)
    train_acc = torch.tensor(0)
    val_loss = torch.tensor(0)
    val_acc = torch.tensor(0)
    test_loss = torch.tensor(0)
    test_acc = torch.tensor(0)
    train_loss_backup = []
    train_acc_backup = []
    val_loss_backup = []
    val_acc_backup = []
    test_loss_backup = []
    test_acc_backup = []

    pbar = tqdm(total=epochs)

    model.to(device)

    for i in range(epochs):
        pbar.set_description(
            f'Epoch{i+1}|tr_loss:{train_loss:.4f}|tr_acc:{train_acc:.4f}|va_loss:{val_loss:.4f}|va_acc:{val_acc:.4f}|te_loss:{test_loss:.4f}|te_acc:{test_acc:.4f}'
        )
        model, train_loss, train_acc = train_loop(train_loader, model, loss_fn,
                                                  optimizer)
        if i == 0:
            val_loss, val_acc = test_loop(val_loader, model, loss_fn)
            test_loss, test_acc = test_loop(test_loader, model, loss_fn)

        if epochs > 10:
            if i % 10 == 0:
                test_loss, test_acc = test_loop(test_loader, model, loss_fn)
            else:
                val_loss, val_acc = test_loop(val_loader, model, loss_fn)

        # scheduler.step(val_loss)
        train_loss_backup.append(train_loss)
        train_acc_backup.append(train_acc)
        test_loss_backup.append(test_loss)
        test_acc_backup.append(test_acc)
        val_loss_backup.append(val_loss)
        val_acc_backup.append(val_acc)
        pbar.update(1)

    print("Done!")

    if plot:
        plt.subplot(1, 2, 1)
        plt.plot(range(epochs), train_loss_backup, label="train_loss")
        plt.plot(range(epochs), val_loss_backup, label="val_loss")
        plt.plot(range(epochs), test_loss_backup, label="test_loss")

        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(range(epochs), train_acc_backup, label="train_acc")
        plt.plot(range(epochs), val_acc_backup, label="val_acc")
        plt.plot(range(epochs), test_acc_backup, label="test_acc")
        plt.legend()
        plt.show()
        confusematrixtest(test_loader, model, loss_fn, class_name)

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


def getmodel(model, path_to_model, device='auto'):
    model.load_state_dict(
        torch.load(path_to_model, map_location=torch.device('cpu')))
    if device == 'auto':
            model.device = ("cuda" if torch.cuda.is_available() else
                           "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        model.device = device
    model.to(model.device)
    model.eval()
    return model


def predict(model, X):
    return model(torch.from_numpy(X).to(torch.float32).to(model.device))