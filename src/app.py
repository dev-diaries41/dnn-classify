from flask import Flask, request, jsonify
from predict import predict_category  # Import the predict_category function from predict.py

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        article_title = request.json['article_title']
        if not article_title:
            return jsonify({'error': 'No input provided.'}), 400
        predicted_category = predict_category(article_title) 
        return jsonify({'predicted_category': predicted_category})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
