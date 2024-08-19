import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# in_features = number of input features
# hidden1 = number of neurons in hidden layer
# out_features = number of categories in classification task
class DNN(nn.Module):
    def __init__(self, in_features, hidden1, hidden2, out_classes):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, out_classes)

        
# defines the forward pass of your neural network. 
# it specifies how the input data flows through the layers of the network to produce the output. 
# it takes an input x and passes it through a single hidden layer and then output layer
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


def save_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(model, filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file '{filepath}' doesn't exist. Create it by running train_model.py.")
    
    # Load the saved model state dictionary into the model
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set the model to evaluation mode
    