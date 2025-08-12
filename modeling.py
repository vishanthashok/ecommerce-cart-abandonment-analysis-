import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

FEATURE_DATA_PATH = "data/processed/ecommerce_features.csv"

def load_data(path):
    return pd.read_csv(path)

if __name__ == "__main__":
    df = load_data(FEATURE_DATA_PATH)

    # Drop non-numeric columns for modeling
    features = df.select_dtypes(include=['number']).drop(columns=['purchased'])
    target = df['purchased']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("📊 Model Evaluation:")
    print(classification_report(y_test, preds))
