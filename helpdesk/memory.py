import sqlite3
import time
from contextlib import closing
from pathlib import Path


class UserMemory:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS preferences ("
                "user_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, "
                "updated_at REAL NOT NULL, PRIMARY KEY (user_id, key))"
            )

    def get_all(self, user_id):
        with closing(sqlite3.connect(self.path)) as conn:
            rows = conn.execute(
                "SELECT key, value FROM preferences WHERE user_id = ?", (user_id,)
            ).fetchall()
        return {key: value for key, value in rows}

    def set(self, user_id, key, value):
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute(
                "INSERT INTO preferences (user_id, key, value, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value, "
                "updated_at = excluded.updated_at",
                (user_id, key, value, time.time()),
            )
