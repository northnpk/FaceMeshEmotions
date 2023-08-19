import torch
from torch import nn
from torch_geometric import nn as geo_nn
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.utils import class_weight
from evalplot import conf_plot, print_eval


class CustomDataset(Dataset):

    def __init__(self, dataframe, device='auto'):
        self.len = len(dataframe)
        feature = dataframe['feature'].values
        edge_index = dataframe['edge_index'][0]
        target = dataframe['target'].values.astype(np.int64)
        if device == 'auto':
            self.device = ("cuda" if torch.cuda.is_available() else "mps"
                           if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
        X = torch.from_numpy(np.array([np.array(f)
                                       for f in feature])).to(torch.float32)

        edge_index = torch.from_numpy(
            np.array(edge_index)).t().contiguous().to(torch.long)

        y = torch.from_numpy(target).to(torch.long)

        self.data = [
            Data(x=X[i], edge_index=edge_index, y=y[i]).to(self.device)
            for i in range(self.len)
        ]

        classweight = class_weight.compute_class_weight(
            class_weight='balanced', classes=np.unique(target), y=target)
        print(f'Class weight:{classweight}')
        self.class_weights = torch.tensor(classweight,
                                          dtype=torch.float).to(self.device)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class GCNClassifier(nn.Module):

    def __init__(self, input_size, output_size, dropout=0.2, device='auto'):
        super().__init__()
        self.dropout_p = dropout
        self.GCNstack = geo_nn.Sequential('x, edge_index, batch', [
            (nn.Dropout(self.dropout_p), 'x -> x'),
            (geo_nn.GCNConv(input_size, 64), 'x, edge_index -> x1'),
            nn.ReLU(inplace=True),
            (geo_nn.GCNConv(64, 64), 'x1, edge_index -> x2'),
            nn.ReLU(inplace=True),
            (lambda x1, x2: [x1, x2], 'x1, x2 -> xs'),
            (geo_nn.JumpingKnowledge("cat", 64, num_layers=2), 'xs -> x'),
            (geo_nn.global_mean_pool, 'x, batch -> x'),
            nn.Linear(2 * 64, output_size),
        ])
        if device == 'auto':
            self.device = ("cuda" if torch.cuda.is_available() else "mps"
                           if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device

        print(f"Using {self.device} device")

    def forward(self, x, edge_index, batch):
        x = self.GCNstack(x, edge_index, batch)
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
            correct += (pred.argmax(1) == data.y).type(
                torch.float).sum().item()
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
            correct += (pred.argmax(1) == data.y).type(
                torch.float).sum().item()
            size += len(data.y)
            pred_list.extend(pred.argmax(1).cpu())
            y_list.extend(data.y.cpu())

    test_loss /= num_batches
    correct /= size
    # print(f'Preds:{np.array(pred_list)}')
    # print(f'y:{np.array(y_list)}')
    conf_plot(y_list, pred_list, class_name)


def trainmodel(model,
               train_df,
               val_df,
               test_df,
               epochs=10,
               lr=1e-4,
               batch_size=8,
               plot=False,
               class_name=None,
               weights=False,
               scheduler=None):
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
    if weights:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
    else:
        loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     factor=0.1,
    #     patience=10,
    #     threshold=0.0001,
    #     threshold_mode='rel',
    #     cooldown=0,
    #     min_lr=4e-6,
    #     eps=1e-08)

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

        if scheduler:
            scheduler.step(val_loss)
        # scheduler.step(val_loss)
        train_loss_backup.append(train_loss)
        train_acc_backup.append(train_acc)
        test_loss_backup.append(test_loss)
        test_acc_backup.append(test_acc)
        val_loss_backup.append(val_loss)
        val_acc_backup.append(val_acc)
        pbar.update(1)
    pbar.close()
    print("Done!")

    if plot:
        print_eval(epochs, (train_loss_backup, train_acc_backup),
                   (test_loss_backup, test_acc_backup),
                   (val_loss_backup, val_acc_backup))
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


def predict(model, X, edge_index, batch=1):
    return model(
        torch.from_numpy(np.array(X)).to(torch.float32).to(model.device),
        torch.from_numpy(np.array(edge_index)).t().contiguous().to(
            torch.long).to(model.device),
        batch.to(torch.long).to(model.device))
