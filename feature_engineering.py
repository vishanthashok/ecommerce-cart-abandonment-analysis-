import pandas as pd
import os

PROCESSED_DATA_PATH = "data/processed/ecommerce_cleaned.csv"
FEATURE_DATA_PATH = "data/processed/ecommerce_features.csv"

def load_data(path):
    return pd.read_csv(path)

def engineer_features(df):
    # Sort by session and time
    df = df.sort_values(by=['user_session', 'event_time'])

    # Create binary target: abandoned (1) if carted but not purchased
    session_purchase = df.groupby('user_session')['event_type'].apply(lambda x: 1 if 'purchase' in x.values else 0)
    df = df.merge(session_purchase.rename('purchased'), on='user_session', how='left')

    # Time spent in session
    session_duration = df.groupby('user_session')['event_time'].apply(lambda x: pd.to_datetime(x).max() - pd.to_datetime(x).min())
    df = df.merge(session_duration.rename('session_duration'), on='user_session', how='left')

    # Price-based features
    df['cart_value'] = df.groupby('user_session')['price'].transform('sum')

    return df

def save_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

if __name__ == "__main__":
    df = load_data(PROCESSED_DATA_PATH)
    feature_df = engineer_features(df)
    save_data(feature_df, FEATURE_DATA_PATH)
    print(f"✅ Features saved to {FEATURE_DATA_PATH}")
