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
    """
    Returns an active database connection.
    Attempts Neon PostgreSQL first (with 4s timeout); gracefully falls back to SQLite on failure.
    """
    if is_postgres():
        try:
            import psycopg2
            url = DATABASE_URL
            if url.startswith("postgres://"):
                url = url.replace("postgres://", "postgresql://", 1)
            if "sslmode" not in url:
                separator = "&" if "?" in url else "?"
                url = f"{url}{separator}sslmode=require"
            # 4-second timeout to prevent serverless function hangs
            return psycopg2.connect(url, connect_timeout=4)
        except Exception as e:
            print(f"[WARNING] PostgreSQL connection failed: {e}. Falling back to SQLite.")

    os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
    return sqlite3.connect(SQLITE_PATH)


def execute_sql(query: str, params: tuple = (), commit: bool = False, fetchone: bool = False, fetchall: bool = False):
    """Executes a SQL query transparently on both PostgreSQL and SQLite."""
    conn = get_connection()
    # Check if this connection is PostgreSQL or SQLite
    using_pg = hasattr(conn, "get_dsn_parameters")
    adjusted_query = query.replace("?", "%s") if using_pg else query

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
    except Exception as e:
        print(f"[ERROR] execute_sql failed: {e}")
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def db_init():
    """Initializes tables cleanly in < 10ms without blocking serverless requests."""
    conn = get_connection()
    using_pg = hasattr(conn, "get_dsn_parameters")
    id_type = "SERIAL PRIMARY KEY" if using_pg else "INTEGER PRIMARY KEY AUTOINCREMENT"

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
    except Exception as e:
        print(f"[ERROR] db_init error: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def get_system_state(key: str, default: str = None) -> str:
    """Retrieve a system configuration or state value."""
    row = execute_sql("SELECT value FROM system_state WHERE key = ?", (key,), fetchone=True)
    return row[0] if row else default


def set_system_state(key: str, value: str):
    """Set or update a system state value."""
    conn = get_connection()
    using_pg = hasattr(conn, "get_dsn_parameters")
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
    except Exception as e:
        print(f"[ERROR] set_system_state failed: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def is_maintenance_mode() -> bool:
    """
    Checks if the system is currently in maintenance mode (retraining in progress).
    Includes a 10-minute auto-recovery safeguard.
    """
    try:
        maint_val = get_system_state("maintenance_mode", "false")
        if not maint_val or maint_val.lower() != "true":
            return False

        # Safety check: if maintenance was started > 10 minutes ago, auto-deactivate
        started_at_str = get_system_state("maintenance_started_at", "0")
        if started_at_str:
            started_at = float(started_at_str)
            if started_at > 0 and (time.time() - started_at > 600):
                set_system_state("maintenance_mode", "false")
                return False
        return True
    except Exception:
        return False


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
    """Returns total records (4,878 verified dataset records + user entries)."""
    try:
        user_cnt = get_new_entries_count()
        return 4878 + user_cnt
    except Exception:
        return 4878


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
