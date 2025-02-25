import pandas as pd
import numpy as np

# Load the data
print("Loading data...")
df = pd.read_csv("Top 100 Crypto Coins/Binance USD.csv")
print("\nFirst few rows of raw data:")
print(df.head())

# Convert date and set index
print("\nConverting date...")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

print("\nData types:")
print(df.dtypes)

# Convert numeric columns
print("\nConverting numeric columns...")
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
df[numeric_columns] = df[numeric_columns].astype(float)

print("\nFinal data types:")
print(df.dtypes)

print("\nFinal data head:")
print(df.head())
