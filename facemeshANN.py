import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as data_utils
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __getitem__(self, index):
        row = self.dataframe.iloc[index].to_numpy()
        features = row[:-1]
        label = row[-1]
        return features, label

    def __len__(self):
        return len(self.dataframe)
    
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

model_path = "./model/facemeshANN.pt"

class ANNClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(124, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
        )

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
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    return model
            
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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return model, test_loss, correct
    
    
def trainmodel(model,train_df, test_df, epochs = 10, learning_rate = 1e-3, batch_size = 10):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    train_loader = DataLoader(CustomDataset(dataframe = train_df))
    test_loader = DataLoader(CustomDataset(dataframe = test_df), shuffle=True)
    
    for i in tqdm(range(epochs)):
        print(f"Epoch {i+1}\n-------------------------------")
        model = train_loop(train_loader, model, loss_fn, optimizer)
        if epochs > 10 :
            if i%(0.1*epochs) == 0:
                model = test_loop(test_loader, model, loss_fn)
            else:
                model, test_loss, correct = test_loop(test_loader, model, loss_fn)
    model, test_loss, correct = test_loop(test_loader, model, loss_fn)
    print("Done!")
    return model, test_loss, correct

def testmodel(model, test_df, batch_size = 1):
    loss_fn = nn.CrossEntropyLoss()
    
    test = data_utils.TensorDataset(test_df['feature'].values, test_df['target'].values)
    test_loader = data_utils.DataLoader(test, batch_size, shuffle=True)
    
    return test_loop(test_loader, model, loss_fn)

def savemodel(model, save_path = './model.pt'):
    torch.save(model, save_path)
    print("saving done!")

def getmodel(path_to_model):
    model = torch.load(path_to_model)
    model.eval()
    return model