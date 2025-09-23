import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / 'data' / 'app.db'

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn