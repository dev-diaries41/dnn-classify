import os
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import joblib

def preprocess(data, category_mapping):
    """
    Preprocess the dataset by converting categorical labels to integers.
    
    Parameters:
        data (pd.DataFrame): The input dataframe.
        category_mapping (dict): A dictionary mapping category names to integers.
    
    Returns:
        pd.DataFrame: The preprocessed dataframe with integer labels.
    """
    data['category'] = data['category'].map(category_mapping)
    return data

def split_dataset(data, input_column, output_column, vectorizer_file_path='data/count_vectorizer.pkl'):
    """
    Split the dataset into training and testing sets and perform text vectorization.
    
    Parameters:
        data (pd.DataFrame): The input dataframe.
        input_column (str): The name of the column containing text data.
        output_column (str): The name of the column containing target labels.
        vectorizer_file_path (str): The file path to save the CountVectorizer object.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test) where X_train and X_test are PyTorch tensors
               representing the feature matrices, and y_train and y_test are PyTorch tensors
               representing the labels.
    """
    # Extract features and labels
    X = data[input_column]
    y = data[output_column]
    
    # Tokenization and Building Vocabulary
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X)
    X = X.toarray()  # Convert sparse matrix to dense array
    
    # Convert labels to integers if they are not already
    if y.dtype == 'object':
        y = y.astype('category').cat.codes
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert feature matrices and labels to PyTorch tensors
    torch.manual_seed(42)
    X_train = torch.Tensor(X_train)
    X_test = torch.Tensor(X_test)
    y_train = torch.LongTensor(y_train.values)
    y_test = torch.LongTensor(y_test.values)
    
    # Save the CountVectorizer object
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