import os
import time
import sqlite3
import pandas as pd

# Support both SQLite and Neon PostgreSQL
DATABASE_URL = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.environ.get("VERCEL"):
    SQLITE_PATH = "/tmp/properties.db"
else:
    SQLITE_PATH = os.path.join(BASE_DIR, "database", "properties.db")

DATA_CSV_PATH = os.path.join(BASE_DIR, "data", "data.csv")


def is_postgres() -> bool:
    """Returns True if a PostgreSQL connection URL is provided."""
    return bool(DATABASE_URL)


def get_connection():
    """Returns an active database connection (Neon PostgreSQL or SQLite)."""
    if is_postgres():
        import psycopg2
        url = DATABASE_URL
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        if "sslmode" not in url:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}sslmode=require"
        return psycopg2.connect(url)
    else:
        os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
        return sqlite3.connect(SQLITE_PATH)


def execute_sql(query: str, params: tuple = (), commit: bool = False, fetchone: bool = False, fetchall: bool = False):
    """Executes a SQL query transparently on both PostgreSQL and SQLite."""
    using_pg = is_postgres()
    adjusted_query = query.replace("?", "%s") if using_pg else query

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(adjusted_query, params)
        result = None
        if fetchone:
            result = cursor.fetchone()
        elif fetchall:
            result = cursor.fetchall()
        if commit:
            conn.commit()
        return result
    finally:
        conn.close()


def db_init():
    """Initializes tables for properties and system state on Neon Postgres or SQLite."""
    using_pg = is_postgres()
    id_type = "SERIAL PRIMARY KEY" if using_pg else "INTEGER PRIMARY KEY AUTOINCREMENT"

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS properties (
                id {id_type},
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

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Migration check for SQLite
        if not using_pg:
            cursor.execute("PRAGMA table_info(properties)")
            columns = [col[1] for col in cursor.fetchall()]
            if "source" not in columns:
                cursor.execute("ALTER TABLE properties ADD COLUMN source TEXT NOT NULL DEFAULT 'user_entry'")

        conn.commit()
    finally:
        conn.close()

    seed_dataset_if_empty()


def get_system_state(key: str, default: str = None) -> str:
    """Retrieve a system configuration or state value."""
    row = execute_sql("SELECT value FROM system_state WHERE key = ?", (key,), fetchone=True)
    return row[0] if row else default


def set_system_state(key: str, value: str):
    """Set or update a system state value."""
    using_pg = is_postgres()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        if using_pg:
            cursor.execute("""
                INSERT INTO system_state (key, value, updated_at)
                VALUES (%s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = CURRENT_TIMESTAMP
            """, (key, str(value)))
        else:
            cursor.execute("""
                INSERT INTO system_state (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP
            """, (key, str(value)))
        conn.commit()
    finally:
        conn.close()


def is_maintenance_mode() -> bool:
    """
    Checks if the system is currently in maintenance mode (retraining in progress).
    Includes a 10-minute auto-recovery safeguard.
    """
    maint_val = get_system_state("maintenance_mode", "false")
    if maint_val.lower() != "true":
        return False

    # Safety check: if maintenance was started > 10 minutes ago, auto-deactivate
    started_at_str = get_system_state("maintenance_started_at", "0")
    try:
        started_at = float(started_at_str)
        if started_at > 0 and (time.time() - started_at > 600):  # 10 minutes
            set_system_state("maintenance_mode", "false")
            return False
    except ValueError:
        pass

    return True


def filter_quality_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filters dataset for age <= 80 and removes price-per-sqft fluctuation spikes."""
    clean_df = df.copy()
    clean_df = clean_df[(clean_df['age'] >= 0) & (clean_df['age'] <= 80)]
    clean_df = clean_df[
        (clean_df['size'] >= 200) & (clean_df['size'] <= 8000) &
        (clean_df['rooms'] >= 1) & (clean_df['rooms'] <= 10) &
        (clean_df['bathrooms'] >= 1) & (clean_df['bathrooms'] <= 6) &
        (clean_df['distance'] >= 0.1) & (clean_df['distance'] <= 50.0)
    ]
    price_per_sqft = clean_df['price'] / clean_df['size']
    q25 = price_per_sqft.quantile(0.10)
    q75 = price_per_sqft.quantile(0.90)
    iqr = q75 - q25
    lower_bound = max(100, q25 - 1.5 * iqr)
    upper_bound = q75 + 1.5 * iqr
    clean_df = clean_df[(price_per_sqft >= lower_bound) & (price_per_sqft <= upper_bound)]
    return clean_df.reset_index(drop=True)


def seed_dataset_if_empty() -> int:
    """Seeds verified dataset into DB if not present."""
    if not os.path.exists(DATA_CSV_PATH):
        return 0

    count_row = execute_sql("SELECT COUNT(*) FROM properties WHERE source = 'dataset'", fetchone=True)
    if count_row and count_row[0] > 0:
        return count_row[0]

    df = pd.read_csv(DATA_CSV_PATH)
    filtered = filter_quality_data(df)

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

    using_pg = is_postgres()
    placeholder = "(%s, %s, %s, %s, %s, %s, %s)" if using_pg else "(?, ?, ?, ?, ?, ?, ?)"
    insert_sql = f"""
        INSERT INTO properties (size, rooms, bathrooms, age, distance, price, source)
        VALUES {placeholder}
    """

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.executemany(insert_sql, rows)
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def add_property(size: float, rooms: int, bathrooms: int, age: int, distance: float, price: int, source: str = "user_entry"):
    """Adds a newly evaluated property to the database."""
    execute_sql(
        "INSERT INTO properties (size, rooms, bathrooms, age, distance, price, source) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (size, rooms, bathrooms, age, distance, price, source),
        commit=True
    )


def get_new_entries_count() -> int:
    """Returns number of new user entries collected since last retraining."""
    row = execute_sql("SELECT COUNT(*) FROM properties WHERE source = 'user_entry'", fetchone=True)
    return row[0] if row else 0


def get_total_count() -> int:
    """Returns total records in the database."""
    row = execute_sql("SELECT COUNT(*) FROM properties", fetchone=True)
    return row[0] if row else 0


def get_user_properties(limit: int = 50):
    """Retrieves recent user-entered properties."""
    return execute_sql(
        "SELECT id, size, rooms, bathrooms, age, distance, price, source, created_at FROM properties WHERE source = 'user_entry' ORDER BY id DESC LIMIT ?",
        (limit,),
        fetchall=True
    ) or []


def delete_property(app_id: int):
    """Deletes a property by ID."""
    execute_sql("DELETE FROM properties WHERE id = ?", (app_id,), commit=True)


def clear_user_properties():
    """Clears recent user valuations."""
    execute_sql("DELETE FROM properties WHERE source = 'user_entry'", commit=True)
