# Crypto Trade Classifier

This project implements a deep learning model to classify cryptocurrency trades as "Good" or "Bad" based on historical price data and technical indicators.

## Features

- LSTM-based deep learning model for time series classification
- Technical indicators including RSI, MACD, and Bollinger Bands
- Flask web interface for easy predictions
- Interactive visualization using Plotly
- Real-time trade classification

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python crypto_classifier.py
```

3. Run the web application:
```bash
python app.py
```

## Usage

1. Open your web browser and navigate to `http://localhost:5000`
2. Upload a CSV file containing cryptocurrency price data
3. Click "Analyze" to get trade predictions
4. View the results in the interactive chart

## Model Details

The model uses a sequence of 30 days of data to predict whether the next day's trade will be profitable. It considers the following features:

- Price data (Open, High, Low, Close)
- Volume
- Technical indicators (RSI, MACD, Bollinger Bands)
- Price returns

The deep learning model architecture consists of:
- Two LSTM layers with dropout for sequence processing
- Dense layers for final classification
- Binary cross-entropy loss for optimization

## Data Format

The input CSV file should contain the following columns:
- Date
- Open
- High
- Low
- Close
- Volume
- Currency

## License

This project is open-source and available under the MIT License.
