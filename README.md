# Crypto Trade Classifier

A deep learning-based system that classifies cryptocurrency trades as "Good" or "Bad" using historical price data and technical indicators.

## Overview

This project implements an LSTM-based neural network to analyze cryptocurrency price patterns and predict potentially profitable trades. The system processes historical data through a web interface, allowing users to upload their own trading data and receive predictions.

## Features

- Deep Learning model using LSTM architecture
- Real-time trade classification
- Technical indicator analysis (RSI, MACD, Bollinger Bands)
- Interactive web interface
- Visual representation of predictions
- CSV file upload support

## Technical Architecture

### Data Processing
- Processes historical price data (OHLCV format)
- Calculates technical indicators
- Creates 30-day sequences for analysis
- Normalizes data for model input

### Model Architecture
- Two-layer LSTM neural network
- Dropout layers for regularization
- Binary classification output (Good/Bad trade)
- PyTorch implementation

### Web Application
- Flask backend
- Interactive frontend using jQuery
- Plotly for visualization
- RESTful API endpoints

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mustansirbharmal-portfolio/Crypto-Trade-Classifier---Assignment.git
cd Crypto-Trade-Classifier---Assignment
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Prepare your data in CSV format with columns:
   - Date
   - Open
   - High
   - Low
   - Close
   - Volume

2. Upload the CSV file through the web interface

3. View the predictions in the interactive chart

## Model Performance

The model analyzes 30-day sequences of market data to predict trade outcomes. While achieving ~44% accuracy, the model provides valuable insights when combined with other trading strategies and proper risk management.

## Dependencies

See `requirements.txt` for complete list of dependencies.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Mustansir Bharmal - [GitHub Profile](https://github.com/mustansirbharmal-portfolio)

Project Link: [https://github.com/mustansirbharmal-portfolio/Crypto-Trade-Classifier---Assignment](https://github.com/mustansirbharmal-portfolio/Crypto-Trade-Classifier---Assignment)
