import os
import sys
import time
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

import db

DATA_CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "property_model.joblib")


def pull_new_data_from_db():
    """Pulls verified user-submitted properties from database and appends to CSV."""
    db.db_init()
    new_rows = db.execute_sql(
        "SELECT size, rooms, bathrooms, age, distance, price FROM properties WHERE source = 'user_entry'",
        fetchall=True
    )
    if not new_rows:
        print("No new user entries found in database.")
        return 0

    print(f"Found {len(new_rows)} new user-submitted property entries.")
    new_df = pd.DataFrame(new_rows, columns=["size", "rooms", "bathrooms", "age", "distance", "price"])

    # Load existing CSV
    if os.path.exists(DATA_CSV_PATH):
        existing_df = pd.read_csv(DATA_CSV_PATH)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df = combined_df.drop_duplicates()
    combined_df.to_csv(DATA_CSV_PATH, index=False)
    print(f"Updated {DATA_CSV_PATH} with {len(combined_df)} total records.")

    # Mark entries in DB as retrained so they are not duplicated
    db.execute_sql("UPDATE properties SET source = 'retrained' WHERE source = 'user_entry'", commit=True)
    return len(new_rows)


def run_retraining():
    """Trains the improved ML model on the updated quality-filtered dataset."""
    print("Starting automated ML retraining pipeline...")
    start_time = time.time()

    # Step 1: Sync new DB records if available
    pull_new_data_from_db()

    # Step 2: Load and apply quality filter
    raw_df = pd.read_csv(DATA_CSV_PATH)
    clean_df = db.filter_quality_data(raw_df)
    print(f"Cleaned dataset records: {len(clean_df)} (Filtered out age > 80 and price fluctuation outliers)")

    features = ["size", "rooms", "bathrooms", "age", "distance"]
    X = clean_df[features]
    y = clean_df["price"]

    # Step 3: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Train Gradient Boosting Regressor
    model = GradientBoostingRegressor(
        n_estimators=140,
        learning_rate=0.07,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Step 5: Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Retraining Complete in {time.time() - start_time:.2f}s!")
    print(f"MAE: ${mae:,.2f} | R² Score: {r2:.4f}")

    if r2 < 0.50:
        print("WARNING: Model quality degraded below threshold. Rejecting update.")
        sys.exit(1)

    # Step 6: Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Successfully saved updated model to {MODEL_PATH}")

    # Step 7: Deactivate maintenance mode
    db.set_system_state("maintenance_mode", "false")
    print("Maintenance mode deactivated. System is live with new model!")


if __name__ == "__main__":
    run_retraining()
