from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


def quote_identifier(value: str) -> str:
    """Quote a SQLite identifier without allowing SQL fragments."""
    return '"' + value.replace('"', '""') + '"'


def sqlite_connection_readonly(path: Path) -> sqlite3.Connection:
    resolved = path.expanduser().resolve(strict=True)
    return sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)


def sqlite_columns(path: Path, table: str) -> list[str]:
    with sqlite_connection_readonly(path) as connection:
        rows = connection.execute(f"PRAGMA table_info({quote_identifier(table)})").fetchall()
    if not rows:
        raise ValueError(f"SQLite table not found or has no columns: {table}")
    return [str(row[1]) for row in rows]


def load_table(path: Path, columns: list[str], table: str = "reports") -> pd.DataFrame:
    path = path.expanduser().resolve(strict=True)
    suffix = path.suffix.lower()
    if suffix in {".db", ".sqlite", ".sqlite3"}:
        available = set(sqlite_columns(path, table))
        missing = sorted(set(columns) - available)
        if missing:
            raise ValueError(f"Missing columns in {table}: {missing}")
        selection = ", ".join(quote_identifier(column) for column in columns)
        with sqlite_connection_readonly(path) as connection:
            return pd.read_sql_query(
                f"SELECT {selection} FROM {quote_identifier(table)}", connection
            )
    if suffix == ".csv":
        return pd.read_csv(path, usecols=columns)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, usecols=columns)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=columns)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
