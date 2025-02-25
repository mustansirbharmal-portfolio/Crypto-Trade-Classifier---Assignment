from flask import Flask, render_template, request, jsonify
import pandas as pd
from crypto_classifier import CryptoTradeClassifier
import json

app = Flask(__name__)
classifier = CryptoTradeClassifier()

try:
    classifier.load_model()
except Exception as e:
    print(f"No pre-trained model found. Error: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Convert data to DataFrame and ensure numeric types
        df = pd.DataFrame(data)
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[numeric_columns] = df[numeric_columns].astype(float)
        
        # Set the date index
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Make predictions
        predictions = classifier.predict(df)
        
        # Convert predictions to list format
        results = [{'date': str(df.index[i]), 'prediction': float(pred[0])} 
                  for i, pred in enumerate(predictions)]
        
        return jsonify({'success': True, 'predictions': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
