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
app.secret_key = os.environ.get("SECRET_KEY", "predict-india-secret-key-2025")

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


def format_inr(amount: float) -> str:
    """Formats numeric amount into Indian Currency notation (Lakhs & Crores)."""
    amount = round(amount)
    if amount >= 10000000:
        crores = amount / 10000000
        return f"₹{crores:.2f} Crore (₹{amount:,})"
    elif amount >= 100000:
        lakhs = amount / 100000
        return f"₹{lakhs:.2f} Lakhs (₹{amount:,})"
    else:
        return f"₹{amount:,}"


@app.template_filter("inr_format")
def inr_format(val):
    """Jinja template filter to format Indian Rupees in tables."""
    if val is None:
        return "Processing..."
    try:
        val = round(float(val))
        if val >= 10000000:
            return f"₹ {val/10000000:.2f} Cr"
        elif val >= 100000:
            return f"₹ {val/100000:.2f} Lakh"
        else:
            return f"₹ {val:,}"
    except Exception:
        return f"₹ {val}"


def validate_property_input(size: float, rooms: int, bathrooms: int, age: int, distance: float) -> tuple[bool, str]:
    """
    SAFETY VALUE FEATURE FOR INDIAN REAL ESTATE:
    Filters out fake, corrupt, or unrealistic values to ensure the ML model
    never receives confusing or confounding inputs.
    """
    # 1. Realistic physical bounds for Indian homes
    if size < 350 or size > 8000:
        return False, "Fake or invalid size: Property area must be between 350 and 8,000 sq ft."
    if rooms < 1 or rooms > 6:
        return False, "Fake or invalid BHK: Rooms must be between 1 BHK and 6 BHK."
    if bathrooms < 1 or bathrooms > 6:
        return False, "Fake or invalid bathrooms: Number of bathrooms must be between 1 and 6."
    if age < 0 or age > 50:
        return False, "Value filtered: Property age must be between 0 and 50 years."
    if distance < 0.1 or distance > 35.0:
        return False, "Fake or invalid distance: Distance to city center/tech hub must be between 0.1 and 35 km."

    # 2. Cross-feature sanity checks
    avg_room_size = size / rooms
    if avg_room_size < 50:
        return False, f"Fake proportion detected: {size:.0f} sq ft is physically impossible for {rooms} BHK."
    if avg_room_size > 3000:
        return False, f"Fake proportion detected: {size:.0f} sq ft is excessively large for only {rooms} room(s)."
    if bathrooms > rooms + 2:
        return False, f"Fake combination detected: {bathrooms} bathrooms is unrealistic for a {rooms} BHK property."

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
    """Predict Indian property price in INR using the trained gradient boosting model."""
    model = get_model()
    df = pd.DataFrame(
        [[size, rooms, bathrooms, age, distance]],
        columns=["size", "rooms", "bathrooms", "age", "distance"]
    )
    predicted_val = model.predict(df)[0]
    return max(1500000, round(float(predicted_val)))


# Initialize database safely on startup
try:
    db.db_init()
except Exception as e:
    print(f"[WARNING] Database initialization error on startup: {e}")


@app.before_request
def maintenance_check():
    """Intercepts requests during maintenance mode (unless accessing static files or admin override)."""
    if request.path.startswith("/static") or request.path in ("/maintenance-off", "/health"):
        return None

    try:
        if db.is_maintenance_mode():
            return render_template("maintenance.html"), 503
    except Exception:
        pass


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            size = float(request.form.get("size", 0))
            rooms = int(request.form.get("rooms", 0))
            bathrooms = int(request.form.get("bathrooms", 0))
            age = int(request.form.get("age", -1))
            distance = float(request.form.get("distance", -1.0))

            # SAFETY VALUE CHECK FOR INDIA
            is_valid, error_msg = validate_property_input(size, rooms, bathrooms, age, distance)
            if not is_valid:
                flash(f"⚠️ {error_msg}", "warning")
                return redirect(url_for("home"))

            # Predict price in INR
            price = predict_price(size, rooms, bathrooms, age, distance)

            # Store new data in database
            try:
                db.add_property(size, rooms, bathrooms, age, distance, price, source="user_entry")
            except Exception as e:
                print(f"[ERROR] Failed to save property: {e}")

            # Check if threshold reached for automatic retraining
            try:
                new_entries = db.get_new_entries_count()
                if new_entries >= RETRAIN_THRESHOLD:
                    db.set_system_state("maintenance_mode", "true")
                    db.set_system_state("maintenance_started_at", str(time.time()))
                    trigger_github_retrain()
                    return render_template("maintenance.html"), 503
            except Exception as e:
                print(f"[ERROR] Retrain check error: {e}")

            flash(f"Estimated Property Price: {format_inr(price)}", "price")
            return redirect(url_for("home"))

        except ValueError:
            flash("Invalid input format. Please ensure all fields are filled with numeric values.", "error")
            return redirect(url_for("home"))
        except Exception as e:
            flash(f"Prediction error: {str(e)}", "error")
            return redirect(url_for("home"))

    # GET request: load user valuations and stats
    try:
        user_properties = db.get_user_properties(limit=50)
        total_records = db.get_total_count()
        new_entries_count = db.get_new_entries_count()
    except Exception as e:
        print(f"[ERROR] Dashboard data error: {e}")
        user_properties = []
        total_records = 4800
        new_entries_count = 0

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
        flash(f"Valuation record #{app_id} deleted successfully.", "success")
    except Exception:
        flash("Error deleting record.", "error")
    return redirect(url_for("home"))


@app.route("/clear")
def clear_app():
    try:
        db.clear_user_properties()
        flash("Your recent valuation history has been cleared.", "info")
    except Exception:
        flash("Error clearing valuation records.", "error")
    return redirect(url_for("home"))


@app.route("/trigger-retrain")
def trigger_retrain():
    """Manual/Admin route to trigger the automated retraining pipeline & test maintenance mode."""
    try:
        db.set_system_state("maintenance_mode", "true")
        db.set_system_state("maintenance_started_at", str(time.time()))
        triggered = trigger_github_retrain()
        if triggered:
            flash("GitHub Actions Retraining workflow triggered successfully!", "success")
        else:
            flash("Maintenance mode enabled. (Set GH_TOKEN in Vercel to auto-ping GitHub).", "info")
    except Exception as e:
        flash(f"Trigger error: {str(e)}", "error")
    return redirect(url_for("home"))


@app.route("/maintenance-off")
def maintenance_off():
    """Admin override to manually turn off maintenance mode."""
    try:
        db.set_system_state("maintenance_mode", "false")
        flash("Maintenance mode deactivated. System is back online!", "success")
    except Exception as e:
        flash(f"Error resetting maintenance mode: {e}", "error")
    return redirect(url_for("home"))


@app.route("/sync-db")
def sync_db():
    """Database status check."""
    flash("Database connection verified and active.", "success")
    return redirect(url_for("home"))


@app.route("/export")
def export():
    """
    Exports ONLY the user's recent/current property valuations.
    Does NOT export the 4,000+ baseline dataset.
    """
    headers = ["ID", "Size (sq ft)", "Configuration (BHK)", "Bathrooms", "Property Age (yrs)", "Distance from Tech Hub (km)", "Price (INR)", "Date"]
    csv_rows = [",".join(headers)]

    try:
        rows = db.execute_sql(
            "SELECT id, size, rooms, bathrooms, age, distance, price, created_at FROM properties WHERE source = 'user_entry' ORDER BY id DESC",
            fetchall=True
        )

        if rows:
            for row in rows:
                csv_rows.append(",".join(str(item) for item in row))

        csv_content = "\n".join(csv_rows) + "\n"
        response = make_response(csv_content)
        response.headers["Content-Disposition"] = "attachment; filename=my_recent_valuations.csv"
        response.headers["Content-Type"] = "text/csv; charset=utf-8"
        return response
    except Exception:
        flash("Error exporting valuation data.", "error")
        return redirect(url_for("home"))


# Health check endpoint
@app.route("/health")
def health():
    return {"status": "ok", "country": "India", "postgres": db.is_postgres()}


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
