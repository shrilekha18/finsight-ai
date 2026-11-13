"""
FinSight AI - SQLite Database Layer
Handles relational data persistence for parsed bank transactions and conversation records.
"""
import sqlite3
from typing import List, Dict, Any, Optional
import pandas as pd
from config import DATABASE_PATH

def get_connection() -> sqlite3.Connection:
    """Returns a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes SQLite schema for transactions and chat history."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Transactions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            description TEXT NOT NULL,
            amount REAL NOT NULL,
            type TEXT NOT NULL CHECK(type IN ('credit', 'debit')),
            category TEXT NOT NULL,
            balance REAL DEFAULT 0.0,
            source_file TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Indexes for rapid queries and aggregations
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tx_date ON transactions(date);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tx_category ON transactions(category);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tx_type ON transactions(type);")
    
    # Chat history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
            content TEXT NOT NULL,
            intent TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()

def insert_transactions(df: pd.DataFrame, source_file: str = "statement") -> int:
    """
    Inserts a DataFrame of transactions into SQLite.
    Expected columns: date, description, amount, type, category, balance (optional).
    Returns count of inserted rows.
    """
    if df.empty:
        return 0
        
    df_clean = df.copy()
    if 'category' not in df_clean.columns:
        df_clean['category'] = 'Miscellaneous'
    if 'balance' not in df_clean.columns:
        df_clean['balance'] = 0.0
    if 'source_file' not in df_clean.columns:
        df_clean['source_file'] = source_file
        
    # Standardize types
    df_clean['amount'] = pd.to_numeric(df_clean['amount'], errors='coerce').fillna(0.0)
    df_clean['balance'] = pd.to_numeric(df_clean['balance'], errors='coerce').fillna(0.0)
    df_clean['type'] = df_clean['type'].astype(str).str.lower().str.strip()
    df_clean['date'] = df_clean['date'].astype(str).str.strip()
    df_clean['description'] = df_clean['description'].astype(str).str.strip()
    df_clean['category'] = df_clean['category'].astype(str).str.strip()
    
    records = []
    for _, row in df_clean.iterrows():
        records.append((
            row['date'],
            row['description'],
            float(row['amount']),
            row['type'],
            row['category'],
            float(row['balance']),
            row['source_file']
        ))
        
    conn = get_connection()
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT INTO transactions (date, description, amount, type, category, balance, source_file)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, records)
    inserted_count = cursor.rowcount
    conn.commit()
    conn.close()
    return inserted_count

def get_all_transactions() -> pd.DataFrame:
    """Retrieves all transactions as a Pandas DataFrame sorted by date descending."""
    conn = get_connection()
    query = """
        SELECT id, date, description, amount, type, category, balance, source_file, created_at
        FROM transactions
        ORDER BY date DESC, id DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_filtered_transactions(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
    tx_type: Optional[str] = None
) -> pd.DataFrame:
    """Queries transactions with optional filters."""
    conn = get_connection()
    query = "SELECT * FROM transactions WHERE 1=1"
    params = []
    
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
    if category and category != "All":
        query += " AND category = ?"
        params.append(category)
    if tx_type and tx_type != "All":
        query += " AND type = ?"
        params.append(tx_type.lower())
        
    query += " ORDER BY date DESC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def clear_all_transactions():
    """Wipes all transactions from the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM transactions")
    conn.commit()
    conn.close()

def save_chat_message(role: str, content: str, intent: Optional[str] = None):
    """Saves a single conversation turn in chat_history."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO chat_history (role, content, intent)
        VALUES (?, ?, ?)
    """, (role, content, intent))
    conn.commit()
    conn.close()

def get_chat_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves recent conversation history."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, role, content, intent, timestamp
        FROM chat_history
        ORDER BY id ASC
        LIMIT ?
    """, (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def clear_chat_history():
    """Wipes chat history from the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_history")
    conn.commit()
    conn.close()

# Auto-initialize database schema when imported
init_db()

