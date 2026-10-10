import os
import sqlite3
import joblib
import pandas as pd
from flask import Flask, redirect, url_for, request, render_template, flash, make_response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)
app.secret_key = os.environ.get("SECRET_KEY", "predict-secret-key-vercel-deployment-2025")

# Setup SQLite database path (use /tmp on Vercel serverless environment)
if os.environ.get("VERCEL"):
    DB_PATH = "/tmp/properties.db"
else:
    DB_PATH = os.path.join(BASE_DIR, "database", "properties.db")

MODEL_PATH = os.path.join(BASE_DIR, "models", "property_model.joblib")
DATA_CSV_PATH = os.path.join(BASE_DIR, "data", "data.csv")
_loaded_model = None


def get_model():
    """Load model with lazy caching."""
    global _loaded_model
    if _loaded_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run train.py first.")
        _loaded_model = joblib.load(MODEL_PATH)
    return _loaded_model


def validate_property_input(size: float, rooms: int, bathrooms: int, age: int, distance: float) -> tuple[bool, str]:
    """
    SAFETY VALUE FEATURE:
    Filters out fake, corrupt, or unrealistic values to ensure the ML model
    never receives confusing or confounding inputs.
    """
    # 1. Realistic physical bounds
    if size < 200 or size > 15000:
        return False, "Fake or invalid size: Property area must be between 200 and 15,000 sq ft."
    if rooms < 1 or rooms > 15:
        return False, "Fake or invalid rooms: Number of rooms must be between 1 and 15."
    if bathrooms < 1 or bathrooms > 10:
        return False, "Fake or invalid bathrooms: Number of bathrooms must be between 1 and 10."
    if age < 0 or age > 100:
        return False, "Value filtered: Property age must be between 0 and 100 years. Properties over 100 years fluctuate excessively."
    if distance < 0.1 or distance > 70.0:
        return False, "Fake or invalid distance: Distance to city center must be between 0.1 and 70 km."

    # 2. Cross-feature sanity checks (catches fake combinations)
    avg_room_size = size / rooms
    if avg_room_size < 50:
        return False, f"Fake proportion detected: {size:.0f} sq ft is too tiny for {rooms} rooms (at least 50 sq ft/room required)."
    if avg_room_size > 4000:
        return False, f"Fake proportion detected: {size:.0f} sq ft is excessively large for only {rooms} room(s)."
    if bathrooms > rooms + 2:
        return False, f"Fake combination detected: {bathrooms} bathrooms is unrealistic for {rooms} room(s)."

    return True, ""


def filter_quality_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    DATA QUALITY FILTER:
    Filters out values that are 'too old' and 'too fluctuated' before saving to DB or training.
    """
    clean_df = df.copy()

    # 1. Filter out properties that are too old (over 80 years)
    clean_df = clean_df[(clean_df['age'] >= 0) & (clean_df['age'] <= 80)]

    # 2. Filter physical sanity bounds
    clean_df = clean_df[
        (clean_df['size'] >= 200) & (clean_df['size'] <= 8000) &
        (clean_df['rooms'] >= 1) & (clean_df['rooms'] <= 10) &
        (clean_df['bathrooms'] >= 1) & (clean_df['bathrooms'] <= 6) &
        (clean_df['distance'] >= 0.1) & (clean_df['distance'] <= 50.0)
    ]

    # 3. Filter high-fluctuation outliers using Price per Sq Ft IQR
    price_per_sqft = clean_df['price'] / clean_df['size']
    q25 = price_per_sqft.quantile(0.10)
    q75 = price_per_sqft.quantile(0.90)
    iqr = q75 - q25
    lower_bound = max(100, q25 - 1.5 * iqr)
    upper_bound = q75 + 1.5 * iqr

    clean_df = clean_df[(price_per_sqft >= lower_bound) & (price_per_sqft <= upper_bound)]
    return clean_df.reset_index(drop=True)


def seed_dataset_to_db() -> int:
    """
    Reads the dataset stored in data/data.csv, filters out too old and fluctuated
    values, and uploads them to the SQLite database.
    """
    if not os.path.exists(DATA_CSV_PATH):
        return 0

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM properties WHERE source = 'dataset'")
        count = cursor.fetchone()[0]
        if count > 0:
            return count  # Already seeded

        df = pd.read_csv(DATA_CSV_PATH)
        filtered = filter_quality_dataset(df)

        rows = [
            (
                float(row['size']),
                int(row['rooms']),
                int(row['bathrooms']),
                int(row['age']),
                float(row['distance']),
                int(row['price']),
                'dataset'
            )
            for _, row in filtered.iterrows()
        ]

        cursor.executemany("""
            INSERT INTO properties (size, rooms, bathrooms, age, distance, price, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, rows)
        conn.commit()
        return len(rows)


def db_init():
    """Initialize database and automatically sync dataset to DB."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS properties (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                size REAL NOT NULL,
                rooms INTEGER NOT NULL,
                bathrooms INTEGER NOT NULL,
                age INTEGER NOT NULL,
                distance REAL NOT NULL,
                price INTEGER NOT NULL,
                source TEXT NOT NULL DEFAULT 'user_entry',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Ensure 'source' column exists if upgrading from older schema
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(properties)")
        columns = [col[1] for col in cursor.fetchall()]
        if "source" not in columns:
            conn.execute("ALTER TABLE properties ADD COLUMN source TEXT NOT NULL DEFAULT 'user_entry'")
        conn.commit()

    # Automatically upload dataset to database with quality filtering
    seed_dataset_to_db()


# Initialize on load
db_init()


def predict_price(size: float, rooms: int, bathrooms: int, age: int, distance: float) -> int:
    """Predict property price using the trained gradient boosting model."""
    model = get_model()
    df = pd.DataFrame(
        [[size, rooms, bathrooms, age, distance]],
        columns=["size", "rooms", "bathrooms", "age", "distance"]
    )
    predicted_val = model.predict(df)[0]
    return max(50000, round(float(predicted_val)))


@app.route("/", methods=["GET", "POST"])
def home():
    db_init()
    if request.method == "POST":
        try:
            size = float(request.form.get("size", 0))
            rooms = int(request.form.get("rooms", 0))
            bathrooms = int(request.form.get("bathrooms", 0))
            age = int(request.form.get("age", -1))
            distance = float(request.form.get("distance", -1.0))

            # SAFETY VALUE CHECK: Filter fake and erratic inputs
            is_valid, error_msg = validate_property_input(size, rooms, bathrooms, age, distance)
            if not is_valid:
                flash(f"⚠️ {error_msg}", "warning")
                return redirect(url_for("home"))

            # Predict price using improved ML model
            price = predict_price(size, rooms, bathrooms, age, distance)

            # Store new data in database with source = 'user_entry'
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute(
                    """INSERT INTO properties (size, rooms, bathrooms, age, distance, price, source)
                       VALUES (?, ?, ?, ?, ?, ?, 'user_entry')""",
                    (size, rooms, bathrooms, age, distance, price)
                )
                conn.commit()

            flash(f"Estimated Property Price: ${price:,.0f}", "price")
            return redirect(url_for("home"))

        except ValueError:
            flash("Invalid input format. Please ensure all fields are filled with numeric values.", "error")
            return redirect(url_for("home"))
        except Exception as e:
            flash(f"Prediction error: {str(e)}", "error")
            return redirect(url_for("home"))

    # GET request: load user valuation history and total database stats
    user_properties = []
    total_db_records = 0
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM properties")
            total_db_records = cursor.fetchone()[0]

            # Fetch recent user-entered valuations
            cursor.execute(
                """SELECT id, size, rooms, bathrooms, age, distance, price, source, created_at
                   FROM properties
                   WHERE source = 'user_entry'
                   ORDER BY id DESC LIMIT 50"""
            )
            user_properties = cursor.fetchall()
    except sqlite3.Error:
        user_properties = []

    return render_template(
        "application.html",
        apps=user_properties,
        total_records=total_db_records
    )


@app.route("/delete/<int:app_id>")
def delete_application(app_id):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM properties WHERE id = ?", (app_id,))
            conn.commit()
        flash(f"Property record #{app_id} deleted successfully.", "success")
    except sqlite3.Error:
        flash("Error deleting property record.", "error")
    return redirect(url_for("home"))


@app.route("/clear")
def clear_app():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            # Clears recent user valuations while preserving the master dataset
            conn.execute("DELETE FROM properties WHERE source = 'user_entry'")
            conn.commit()
        flash("Your recent valuation history has been cleared.", "info")
    except sqlite3.Error:
        flash("Error clearing property records.", "error")
    return redirect(url_for("home"))


@app.route("/sync-db")
def sync_db():
    """Manually trigger dataset upload and quality filtering into the database."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM properties WHERE source = 'dataset'")
            conn.commit()
        count = seed_dataset_to_db()
        flash(f"Database successfully synced: {count:,} high-quality records uploaded (filtered for age & fluctuations).", "success")
    except Exception as e:
        flash(f"Database sync error: {str(e)}", "error")
    return redirect(url_for("home"))


@app.route("/export")
def export():
    export_type = request.args.get("type", "user")
    headers = ["ID", "Size (sq ft)", "Rooms", "Bathrooms", "Property Age (yrs)", "Distance from City (km)", "Price ($)", "Source", "Date"]
    csv_rows = [",".join(headers)]
    try:
        with sqlite3.connect(DB_PATH) as conn:
            if export_type == "all":
                cursor = conn.execute("SELECT id, size, rooms, bathrooms, age, distance, price, source, created_at FROM properties ORDER BY id DESC")
                filename = "all_database_properties.csv"
            else:
                cursor = conn.execute("SELECT id, size, rooms, bathrooms, age, distance, price, source, created_at FROM properties WHERE source = 'user_entry' ORDER BY id DESC")
                filename = "my_property_predictions.csv"

            for row in cursor.fetchall():
                csv_rows.append(",".join(str(item) for item in row))

        csv_content = "\n".join(csv_rows) + "\n"
        response = make_response(csv_content)
        response.headers["Content-Disposition"] = f"attachment; filename={filename}"
        response.headers["Content-Type"] = "text/csv"
        return response
    except sqlite3.Error:
        flash("Error exporting data.", "error")
        return redirect(url_for("home"))


# Redirect legacy routes to home
@app.route("/login")
@app.route("/signup")
@app.route("/logout")
@app.route("/application")
def auth_redirect():
    return redirect(url_for("home"))


if __name__ == "__main__":
    db_init()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
