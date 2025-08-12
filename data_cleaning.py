import pandas as pd
import os

RAW_DATA_PATH = "data/raw/ecommerce_data.csv"
PROCESSED_DATA_PATH = "data/processed/ecommerce_cleaned.csv"

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    # Convert timestamps
    df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce')

    # Drop rows with missing user/session IDs
    df = df.dropna(subset=['user_id', 'user_session'])

    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing brands
    df['brand'] = df['brand'].fillna("Unknown")

    return df

def save_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

if __name__ == "__main__":
    df = load_data(RAW_DATA_PATH)
    cleaned_df = clean_data(df)
    save_data(cleaned_df, PROCESSED_DATA_PATH)
    print(f"✅ Cleaned data saved to {PROCESSED_DATA_PATH}")
