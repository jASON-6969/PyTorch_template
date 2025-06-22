import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import Linear, ReLU, Softmax, Module
from torch.nn.init import kaiming_uniform_, xavier_uniform_
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

# Define the dataset class
class CSVDataset(Dataset):
    def __init__(self, path):
        # Load the CSV file
        df = pd.read_csv(path, header=None)
        # Extract features and labels
        self.X = df.values[:, :-1]  # All feature columns
        self.y = df.values[:, -1]   # Label column
        # Standardize features
        self.X = StandardScaler().fit_transform(self.X)
        # Ensure features are float32
        self.X = self.X.astype('float32')
        # Encode labels
        self.y = LabelEncoder().fit_transform(self.y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]
    
    def get_splits(self, n_test=0.33):
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        return random_split(self, [train_size, test_size])

# Define the MLP model
class MLP(Module):
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # First hidden layer
        self.hidden1 = Linear(n_inputs, 10)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # Second hidden layer
        self.hidden2 = Linear(10, 8)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # Output layer
        self.hidden3 = Linear(8, 3)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Softmax(dim=1)
    
    def forward(self, X):
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.hidden3(X)
        X = self.act3(X)
        return X

# Main function
def main():
    # Set dataset path
    data_path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
    
    # Create dataset
    dataset = CSVDataset(data_path)
    print(f'Input matrix shape: {dataset.X.shape}')
    print(f'Output matrix shape: {dataset.y.shape}')
    print(f'Total dataset length: {len(dataset)}')
    
    # Split into training and test sets
    train, test = dataset.get_splits(n_test=0.33)
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    print(f'Training set length: {len(train)}')
    print(f'Test set length: {len(test)}')
    
    # Initialize model
    n_inputs = dataset.X.shape[1]  # Number of features (4)
    model = MLP(n_inputs)
    
    # Define loss function and optimizer
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Train model
    model.train()
    for epoch in range(500):
        for i, (inputs, targets) in enumerate(train_dl):
            optimizer.zero_grad()
            yhat = model(inputs)
            loss = criterion(yhat, targets)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')
    
    # Test model
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for inputs, targets in test_dl:
            yhat = model(inputs)
            yhat = yhat.detach().numpy()
            actual = targets.numpy()
            yhat = np.argmax(yhat, axis=1)
            actual = actual.reshape((len(actual), 1))
            yhat = yhat.reshape((len(yhat), 1))
            predictions.append(yhat)
            actuals.append(actual)
    
    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)
    acc = accuracy_score(actuals, predictions)
    print(f'Test set accuracy: {acc:.4f}')
    
    # Single sample prediction
    row = [5.1, 3.5, 1.4, 0.2]  # Example data
    row = StandardScaler().fit(dataset.X).transform([row])  # Standardize
    row = Tensor(row)
    model.eval()
    with torch.no_grad():
        yhat = model(row)
        yhat = yhat.detach().numpy()
    print(f'Single sample prediction probabilities: {yhat[0].round(4)} (Most likely class: class={np.argmax(yhat)})')


if __name__ == '__main__':
    main()