"""
FinSight AI - Central Configuration Module
Loads environment variables and sets system-wide paths and constants.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables
load_dotenv(BASE_DIR / ".env")

# LLM Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Database & Vector DB paths
DATABASE_PATH = os.getenv("DATABASE_PATH", str(DATA_DIR / "finsight.db"))
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma_db"))

# Standard Financial Categories
CATEGORIES = [
    "Housing",
    "Utilities",
    "Groceries",
    "Dining Out",
    "Transportation",
    "Healthcare",
    "Entertainment",
    "Shopping",
    "Investment & Savings",
    "Income / Payroll",
    "Miscellaneous"
]

# 50/30/20 Budgeting Classifications
CATEGORY_BUDGET_MAP = {
    "Housing": "Needs",
    "Utilities": "Needs",
    "Groceries": "Needs",
    "Transportation": "Needs",
    "Healthcare": "Needs",
    "Dining Out": "Wants",
    "Entertainment": "Wants",
    "Shopping": "Wants",
    "Investment & Savings": "Savings",
    "Income / Payroll": "Income",
    "Miscellaneous": "Wants"
}

# 50/30/20 Target Ratios
BUDGET_TARGETS = {
    "Needs": 0.50,
    "Wants": 0.30,
    "Savings": 0.20
}

