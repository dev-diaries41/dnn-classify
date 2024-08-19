import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from data import preprocess, split_dataset, plot_test_accuracy, plot_training_loss
from model import DNN, save_model
from sklearn.metrics import accuracy_score


# Load the dataset
# dataset in CSV format with columns "title" and "category"
df = pd.read_csv("data/data.csv")

# Preprocess the data
df = preprocess(df)
X_train, X_test, y_train, y_test = split_dataset(df)

# Define model parameters]
in_features = X_train.shape[1]  # Number of input features
hidden1 = 32
hidden2 = 16  
out = 4  # Number of categories

# Instantiate model
model = DNN(in_features, hidden1, hidden2, out)

learning_rate = 0.0025
num_epochs = 100
criterion = nn.CrossEntropyLoss()   # Define the loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)    # optimizer

def train(model, criterion, optimizer, X_train, y_train, X_test, y_test, num_epochs):
    train_loss_values = []
    test_accuracy_values = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        train_loss = criterion(outputs, y_train)
        train_loss.backward()
        optimizer.step()
        train_loss_values.append(train_loss.item())

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, predicted = torch.max(test_outputs, 1)
            test_accuracy = accuracy_score(y_test, predicted.numpy())
            test_accuracy_values.append(test_accuracy)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Train Loss: {train_loss.item():.4f}, Test Accuracy: {test_accuracy:.4f}')

    return train_loss_values, test_accuracy_values



train_loss_values, test_accuracy_values = train(model, criterion, optimizer, X_train, y_train, X_test, y_test, num_epochs)

plot_training_loss(train_loss_values)
plot_test_accuracy(test_accuracy_values)
save_model(model, 'data/articles_model.pt')

