import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def filter_quality_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out 'too old' properties and extreme 'fluctuating' outlier values
    so the ML model is not distorted by erratic or non-standard real estate data.
    """
    clean_df = df.copy()

    # 1. Filter properties that are too old (antique/heritage properties fluctuate wildly)
    clean_df = clean_df[(clean_df['age'] >= 0) & (clean_df['age'] <= 80)]

    # 2. Filter physical sanity bounds (safety check on dataset)
    clean_df = clean_df[
        (clean_df['size'] >= 200) & (clean_df['size'] <= 8000) &
        (clean_df['rooms'] >= 1) & (clean_df['rooms'] <= 10) &
        (clean_df['bathrooms'] >= 1) & (clean_df['bathrooms'] <= 6) &
        (clean_df['distance'] >= 0.1) & (clean_df['distance'] <= 50.0)
    ]

    # 3. Filter high-fluctuation outliers using Price per Square Foot IQR
    price_per_sqft = clean_df['price'] / clean_df['size']
    q25 = price_per_sqft.quantile(0.10)
    q75 = price_per_sqft.quantile(0.90)
    iqr = q75 - q25
    lower_bound = max(100, q25 - 1.5 * iqr)
    upper_bound = q75 + 1.5 * iqr

    clean_df = clean_df[(price_per_sqft >= lower_bound) & (price_per_sqft <= upper_bound)]

    return clean_df.reset_index(drop=True)


def train_model():
    data_path = 'data/data.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    raw_data = pd.read_csv(data_path)
    print(f"Loaded raw dataset: {len(raw_data)} records")

    # Apply quality filter (too old and high fluctuation filter)
    clean_data = filter_quality_data(raw_data)
    print(f"Dataset after removing too old & fluctuated data: {len(clean_data)} high-quality records")

    features = ['size', 'rooms', 'bathrooms', 'age', 'distance']
    X = clean_data[features]
    y = clean_data['price']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Gradient Boosting Regressor (Non-linear, robust to noise)
    model = GradientBoostingRegressor(
        n_estimators=140,
        learning_rate=0.07,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("--- Model Performance ---")
    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"R² Score: {r2:.4f}")

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/property_model.joblib')
    print("Model saved to models/property_model.joblib")


if __name__ == '__main__':
    train_model()
