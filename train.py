import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def filter_quality_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quality filter tailored for Indian urban residential properties:
    Filters out extreme outliers and erratic fluctuations.
    """
    clean_df = df.copy()

    # 1. Physical sanity bounds for Indian homes
    clean_df = clean_df[
        (clean_df['size'] >= 350) & (clean_df['size'] <= 8000) &
        (clean_df['rooms'] >= 1) & (clean_df['rooms'] <= 8) &
        (clean_df['bathrooms'] >= 1) & (clean_df['bathrooms'] <= 8) &
        (clean_df['age'] >= 0) & (clean_df['age'] <= 60) &
        (clean_df['distance'] >= 0.1) & (clean_df['distance'] <= 50.0) &
        (clean_df['price'] >= 1000000) & (clean_df['price'] <= 80000000)
    ]

    # 2. Filter rate-per-sqft fluctuations (using 5th to 95th percentiles)
    rate_per_sqft = clean_df['price'] / clean_df['size']
    lower_rate = rate_per_sqft.quantile(0.04)
    upper_rate = rate_per_sqft.quantile(0.96)
    clean_df = clean_df[(rate_per_sqft >= lower_rate) & (rate_per_sqft <= upper_rate)]

    return clean_df.reset_index(drop=True)


def train_model():
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'data.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    raw_data = pd.read_csv(data_path)
    clean_data = filter_quality_data(raw_data)
    print(f"Training on {len(clean_data)} verified Indian property records...")

    features = ['size', 'rooms', 'bathrooms', 'age', 'distance']
    X = clean_data[features]
    y = clean_data['price']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Gradient Boosting Regressor
    model = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.07,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("--- Model Performance (India) ---")
    print(f"MAE: Rs. {mae:,.2f} ({mae/100000:.2f} Lakhs)")
    print(f"R2 Score: {r2:.4f}")

    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, 'property_model.joblib'))
    print("Saved Indian real estate model to models/property_model.joblib")


if __name__ == '__main__':
    train_model()
