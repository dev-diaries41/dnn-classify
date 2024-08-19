import torch
from model import DNN, load_model
from data import preprocess_input_inference
import joblib


# Load and return the CountVectorizer object used during training
file_path = "data/articles_model.pt"
vectorizer = joblib.load('data/count_vectorizer.pkl')
print(f"length: {len(vectorizer.get_feature_names_out())}")
    
# Instantiate model
# Load the saved model state dictionary into the model
# Define model parameters

in_features = len(vectorizer.get_feature_names_out())
hidden1 = 32
hidden2 = 16  
out_features = 4  # Number of categories
model = DNN(in_features, hidden1, hidden2, out_features)
load_model(model, file_path)

def predict_category(input_data):
    # Preprocess input data
    preprocessed_input = preprocess_input_inference([input_data], vectorizer)

    # Pass preprocessed input data through the model for inference
    with torch.no_grad():
        output = model(preprocessed_input)
        _, predicted = torch.max(output, 1)

    # Return the predicted category
    categories = {0: 'AI', 1: 'Biology', 2: 'JavaScript', 3: 'Physics'}
    predicted_category = categories[predicted.item()]
    return predicted_category

if __name__ == "__main__":
    # Prompt the user to enter input data
    input_data = input("Enter the article title: ")
    predicted_category = predict_category(input_data)
    print("Predicted category:", predicted_category)
