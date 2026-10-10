import os
import json
import time
import urllib.request
import joblib
import pandas as pd
from flask import Flask, redirect, url_for, request, render_template, flash, make_response

import db

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)
app.secret_key = os.environ.get("SECRET_KEY", "predict-secret-key-vercel-deployment-2025")

MODEL_PATH = os.path.join(BASE_DIR, "models", "property_model.joblib")
RETRAIN_THRESHOLD = int(os.environ.get("RETRAIN_THRESHOLD", 500))

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


def trigger_github_retrain() -> bool:
    """Dispatches a repository_dispatch event to trigger GitHub Actions retraining."""
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY", "rohanrarya/predict")

    if not token:
        print("[WARNING] GH_TOKEN environment variable not set. Cannot trigger GitHub Actions automatically.")
        return False

    url = f"https://api.github.com/repos/{repo}/dispatches"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Predict-Flask-App",
        "Content-Type": "application/json"
    }
    payload = json.dumps({
        "event_type": "retrain_model",
        "client_payload": {
            "timestamp": str(pd.Timestamp.now()),
            "reason": f"{RETRAIN_THRESHOLD}_new_entries_threshold_reached"
        }
    }).encode("utf-8")

    try:
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req) as resp:
            return resp.status in (200, 204)
    except Exception as e:
        print(f"[ERROR] Failed to trigger GitHub Action: {e}")
        return False


def predict_price(size: float, rooms: int, bathrooms: int, age: int, distance: float) -> int:
    """Predict property price using the trained gradient boosting model."""
    model = get_model()
    df = pd.DataFrame(
        [[size, rooms, bathrooms, age, distance]],
        columns=["size", "rooms", "bathrooms", "age", "distance"]
    )
    predicted_val = model.predict(df)[0]
    return max(50000, round(float(predicted_val)))


# Initialize database on startup
db.db_init()


@app.before_request
def maintenance_check():
    """Intercepts requests during maintenance mode (unless accessing static files or admin override)."""
    # Whitelist static files and the maintenance-off admin route
    if request.path.startswith("/static") or request.path in ("/maintenance-off", "/health"):
        return None

    if db.is_maintenance_mode():
        return render_template("maintenance.html"), 503


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            size = float(request.form.get("size", 0))
            rooms = int(request.form.get("rooms", 0))
            bathrooms = int(request.form.get("bathrooms", 0))
            age = int(request.form.get("age", -1))
            distance = float(request.form.get("distance", -1.0))

            # SAFETY VALUE CHECK
            is_valid, error_msg = validate_property_input(size, rooms, bathrooms, age, distance)
            if not is_valid:
                flash(f"⚠️ {error_msg}", "warning")
                return redirect(url_for("home"))

            # Predict price
            price = predict_price(size, rooms, bathrooms, age, distance)

            # Store new data in database
            db.add_property(size, rooms, bathrooms, age, distance, price, source="user_entry")

            # Check if threshold reached for automatic retraining
            new_entries = db.get_new_entries_count()
            if new_entries >= RETRAIN_THRESHOLD:
                # Activate maintenance mode
                db.set_system_state("maintenance_mode", "true")
                db.set_system_state("maintenance_started_at", str(time.time()))
                # Trigger GitHub Actions in background
                trigger_github_retrain()
                return render_template("maintenance.html"), 503

            flash(f"Estimated Property Price: ${price:,.0f}", "price")
            return redirect(url_for("home"))

        except ValueError:
            flash("Invalid input format. Please ensure all fields are filled with numeric values.", "error")
            return redirect(url_for("home"))
        except Exception as e:
            flash(f"Prediction error: {str(e)}", "error")
            return redirect(url_for("home"))

    # GET request: load user valuations and stats
    user_properties = db.get_user_properties(limit=50)
    total_records = db.get_total_count()
    new_entries_count = db.get_new_entries_count()

    return render_template(
        "application.html",
        apps=user_properties,
        total_records=total_records,
        new_entries=new_entries_count,
        threshold=RETRAIN_THRESHOLD,
        is_postgres=db.is_postgres()
    )


@app.route("/delete/<int:app_id>")
def delete_application(app_id):
    try:
        db.delete_property(app_id)
        flash(f"Property record #{app_id} deleted successfully.", "success")
    except Exception:
        flash("Error deleting property record.", "error")
    return redirect(url_for("home"))


@app.route("/clear")
def clear_app():
    try:
        db.clear_user_properties()
        flash("Your recent valuation history has been cleared.", "info")
    except Exception:
        flash("Error clearing property records.", "error")
    return redirect(url_for("home"))


@app.route("/trigger-retrain")
def trigger_retrain():
    """Manual/Admin route to trigger the automated retraining pipeline & test maintenance mode."""
    db.set_system_state("maintenance_mode", "true")
    db.set_system_state("maintenance_started_at", str(time.time()))
    triggered = trigger_github_retrain()
    if triggered:
        flash("GitHub Actions Retraining workflow triggered successfully!", "success")
    else:
        flash("Maintenance mode enabled. (Set GH_TOKEN in Vercel to auto-ping GitHub).", "info")
    return redirect(url_for("home"))


@app.route("/maintenance-off")
def maintenance_off():
    """Admin override to manually turn off maintenance mode."""
    db.set_system_state("maintenance_mode", "false")
    flash("Maintenance mode deactivated. System is back online!", "success")
    return redirect(url_for("home"))


@app.route("/sync-db")
def sync_db():
    """Manually trigger dataset upload and quality filtering into the database."""
    try:
        count = db.seed_dataset_if_empty()
        flash(f"Database synced successfully ({count:,} verified records).", "success")
    except Exception as e:
        flash(f"Database sync error: {str(e)}", "error")
    return redirect(url_for("home"))


@app.route("/export")
def export():
    export_type = request.args.get("type", "user")
    headers = ["ID", "Size (sq ft)", "Rooms", "Bathrooms", "Property Age (yrs)", "Distance from City (km)", "Price ($)", "Source", "Date"]
    csv_rows = [",".join(headers)]

    try:
        if export_type == "all":
            rows = db.execute_sql("SELECT id, size, rooms, bathrooms, age, distance, price, source, created_at FROM properties ORDER BY id DESC", fetchall=True)
            filename = "all_database_properties.csv"
        else:
            rows = db.execute_sql("SELECT id, size, rooms, bathrooms, age, distance, price, source, created_at FROM properties WHERE source = 'user_entry' ORDER BY id DESC", fetchall=True)
            filename = "my_property_predictions.csv"

        for row in rows:
            csv_rows.append(",".join(str(item) for item in row))

        csv_content = "\n".join(csv_rows) + "\n"
        response = make_response(csv_content)
        response.headers["Content-Disposition"] = f"attachment; filename={filename}"
        response.headers["Content-Type"] = "text/csv"
        return response
    except Exception:
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
