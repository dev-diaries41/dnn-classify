import os
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import joblib

def preprocess(data):
    # Change categories from string to integers
    data['category'] = data['category'].replace({'AI': 0, 'Biology': 1, 'JavaScript': 2, 'Physics': 3})
    return data

def split_dataset(data):
    X = data.drop('category', axis=1)['title']  
    
    # Tokenization and Building Vocabulary
    vectorizer = CountVectorizer()

    X = vectorizer.fit_transform(X)
    X = X.toarray()  # X is the feature matrix
    y = data['category']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert feature matrices and labels to PyTorch tensors
    torch.manual_seed(42)
    X_train = torch.Tensor(X_train)
    X_test = torch.Tensor(X_test)
    y_train = torch.LongTensor(y_train.values)  # Convert to PyTorch tensor
    y_test = torch.LongTensor(y_test.values)  # Convert to PyTorch tensor

    # Save the CountVectorizer object
    vectorizer_file_path = 'data/count_vectorizer.pkl'
    os.makedirs(os.path.dirname(vectorizer_file_path), exist_ok=True)
    joblib.dump(vectorizer, vectorizer_file_path)

    return X_train, X_test, y_train, y_test

# Function to preprocess input data using the same tokenizer used during training
def preprocess_input_inference(input_data, vectorizer):
    # Transform input_data using the CountVectorizer
    input_data = vectorizer.transform(input_data)  # X is the feature matrix
    
    # Convert the sparse matrix to a dense numpy array
    input_data = input_data.toarray()
    return torch.Tensor(input_data)

def plot_training_loss(loss_values, save_path='results/training_loss_plot.png'):
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.plot(loss_values, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

def plot_test_accuracy(accuracy_values, save_path='results/test_accuracy_plot.png'):
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.plot(accuracy_values, label='Test Accuracy', color='green')
    plt.title('Epoch vs Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()