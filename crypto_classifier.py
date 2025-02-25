import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import ta
import joblib
import os

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=50):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(25, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x[:, -1, :])  # Take only the last output
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

class CryptoTradeClassifier:
    def __init__(self, lookback=30):
        self.lookback = lookback
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def prepare_data(self, df):
        # Make a copy to avoid modifying the original dataframe
        df = df.copy()
        
        # Drop the Currency column if it exists
        if 'Currency' in df.columns:
            df = df.drop('Currency', axis=1)
        
        # Ensure all numeric columns are float
        df = df.astype(float)
        
        # Add technical indicators
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df['MACD'] = ta.trend.MACD(df['Close']).macd()
        df['BB_high'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
        df['BB_low'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
        
        # Calculate daily returns
        df['Returns'] = df['Close'].pct_change()
        
        # Create labels (1 for positive returns, 0 for negative)
        df['Label'] = (df['Returns'].shift(-1) > 0).astype(int)
        
        # Drop rows with NaN values
        df.dropna(inplace=True)
        
        return df
    
    def create_sequences(self, data, labels):
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i:(i + self.lookback)])
            y.append(labels[i + self.lookback])
        return np.array(X), np.array(y)
    
    def fit(self, df):
        print("Preparing data...")
        df = self.prepare_data(df)
        
        # Select features for training
        features = ['Open', 'High', 'Low', 'Close', 'Volume',
                   'RSI', 'MACD', 'BB_high', 'BB_low', 'Returns']
        
        print("Scaling features...")
        # Scale the features
        scaled_data = self.scaler.fit_transform(df[features])
        labels = df['Label'].values
        
        print("Creating sequences...")
        # Create sequences
        X, y = self.create_sequences(scaled_data, labels)
        
        print(f"Created {len(X)} sequences with shape {X.shape}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create data loaders
        train_dataset = TimeSeriesDataset(X_train, y_train)
        test_dataset = TimeSeriesDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        print("Initializing model...")
        # Initialize the model
        self.model = LSTMModel(input_size=len(features)).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters())
        
        print("Training model...")
        # Training loop
        num_epochs = 50
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')
        
        print("Evaluating model...")
        # Evaluate the model
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = self.model(batch_X)
                predicted = (outputs.squeeze() > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        accuracy = correct / total
        print(f"\nTest accuracy: {accuracy:.4f}")
        
        return accuracy
    
    def predict(self, df):
        # Prepare the data
        df = self.prepare_data(df)
        
        # Select features
        features = ['Open', 'High', 'Low', 'Close', 'Volume',
                   'RSI', 'MACD', 'BB_high', 'BB_low', 'Returns']
        
        # Scale the features
        scaled_data = self.scaler.transform(df[features])
        labels = df['Label'].values
        
        # Create sequences
        X, _ = self.create_sequences(scaled_data, labels)
        
        # Convert to tensor and move to device
        X = torch.FloatTensor(X).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X).cpu().numpy()
        
        return predictions
    
    def save_model(self, model_path='model.pth', scaler_path='scaler.pkl'):
        torch.save(self.model.state_dict(), model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Model saved to {model_path} and {scaler_path}")
    
    def load_model(self, model_path='model.pth', scaler_path='scaler.pkl'):
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = LSTMModel(input_size=10).to(self.device)  # 10 features
            self.model.load_state_dict(torch.load(model_path))
            self.scaler = joblib.load(scaler_path)
            print(f"Model loaded from {model_path} and {scaler_path}")
        else:
            raise FileNotFoundError("Model files not found. Please train the model first.")

if __name__ == "__main__":
    # Load the data
    print("Loading data...")
    df = pd.read_csv("Top 100 Crypto Coins/Binance USD.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Create and train the model
    classifier = CryptoTradeClassifier(lookback=30)
    accuracy = classifier.fit(df)
    
    # Save the model
    classifier.save_model()
