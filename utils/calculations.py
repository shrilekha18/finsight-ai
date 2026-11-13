"""
FinSight AI - Financial Calculation Engine
Provides deterministic, Python-based mathematical aggregations and analytics.
All monetary totals, ratios, and breakdowns are computed here to ensure 100% mathematical accuracy.
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from config import CATEGORY_BUDGET_MAP, BUDGET_TARGETS

def calculate_summary_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes overarching financial health metrics from a transactions DataFrame.
    """
    if df.empty:
        return {
            "total_income": 0.0,
            "total_expenses": 0.0,
            "net_cash_flow": 0.0,
            "savings_rate": 0.0,
            "transaction_count": 0,
            "credit_count": 0,
            "debit_count": 0,
            "avg_expense": 0.0,
            "largest_expense": {"description": "N/A", "amount": 0.0, "date": "N/A"},
            "date_range": ("N/A", "N/A"),
            "daily_burn_rate": 0.0
        }
        
    df_calc = df.copy()
    if 'category' not in df_calc.columns:
        df_calc['category'] = 'Miscellaneous'
    df_calc['amount'] = pd.to_numeric(df_calc['amount'], errors='coerce').fillna(0.0)
    df_calc['type'] = df_calc['type'].astype(str).str.lower().str.strip()
    
    credits = df_calc[df_calc['type'] == 'credit']
    debits = df_calc[df_calc['type'] == 'debit']
    
    total_income = round(float(credits['amount'].sum()), 2)
    total_expenses = round(float(debits['amount'].sum()), 2)
    net_cash_flow = round(total_income - total_expenses, 2)
    
    savings_rate = 0.0
    if total_income > 0:
        savings_rate = round(max(0.0, ((total_income - total_expenses) / total_income) * 100), 2)
        
    # Largest single expense
    if not debits.empty:
        max_debit_row = debits.loc[debits['amount'].idxmax()]
        largest_expense = {
            "description": str(max_debit_row.get('description', 'N/A')),
            "amount": round(float(max_debit_row.get('amount', 0.0)), 2),
            "date": str(max_debit_row.get('date', 'N/A'))
        }
        avg_expense = round(float(debits['amount'].mean()), 2)
    else:
        largest_expense = {"description": "N/A", "amount": 0.0, "date": "N/A"}
        avg_expense = 0.0
        
    # Date range and daily burn
    try:
        dt_series = pd.to_datetime(df_calc['date'], errors='coerce').dropna()
        if not dt_series.empty:
            min_date = dt_series.min().strftime('%Y-%m-%d')
            max_date = dt_series.max().strftime('%Y-%m-%d')
            days_span = max(1, (dt_series.max() - dt_series.min()).days + 1)
            daily_burn_rate = round(total_expenses / days_span, 2)
        else:
            min_date, max_date = "N/A", "N/A"
            daily_burn_rate = 0.0
    except Exception:
        min_date, max_date = "N/A", "N/A"
        daily_burn_rate = 0.0
        
    return {
        "total_income": total_income,
        "total_expenses": total_expenses,
        "net_cash_flow": net_cash_flow,
        "savings_rate": savings_rate,
        "transaction_count": len(df_calc),
        "credit_count": len(credits),
        "debit_count": len(debits),
        "avg_expense": avg_expense,
        "largest_expense": largest_expense,
        "date_range": (min_date, max_date),
        "daily_burn_rate": daily_burn_rate
    }

def calculate_category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups debit transactions by category and computes totals, counts, and percentages.
    """
    if df.empty:
        return pd.DataFrame(columns=['category', 'total_amount', 'transaction_count', 'percentage'])
        
    df_calc = df.copy()
    if 'category' not in df_calc.columns:
        df_calc['category'] = 'Miscellaneous'
    df_calc['amount'] = pd.to_numeric(df_calc['amount'], errors='coerce').fillna(0.0)
    df_calc['type'] = df_calc['type'].astype(str).str.lower().str.strip()
    
    debits = df_calc[df_calc['type'] == 'debit']
    if debits.empty:
        return pd.DataFrame(columns=['category', 'total_amount', 'transaction_count', 'percentage'])
        
    grouped = debits.groupby('category').agg(
        total_amount=('amount', 'sum'),
        transaction_count=('amount', 'count')
    ).reset_index()
    
    total_debits = grouped['total_amount'].sum()
    if total_debits > 0:
        grouped['percentage'] = (grouped['total_amount'] / total_debits * 100).round(2)
    else:
        grouped['percentage'] = 0.0
        
    grouped['total_amount'] = grouped['total_amount'].round(2)
    grouped = grouped.sort_values(by='total_amount', ascending=False).reset_index(drop=True)
    return grouped

def calculate_monthly_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates income, expenses, and net cash flow grouped by calendar month.
    """
    if df.empty:
        return pd.DataFrame(columns=['month', 'income', 'expenses', 'net_savings'])
        
    df_calc = df.copy()
    if 'category' not in df_calc.columns:
        df_calc['category'] = 'Miscellaneous'
    df_calc['amount'] = pd.to_numeric(df_calc['amount'], errors='coerce').fillna(0.0)
    df_calc['type'] = df_calc['type'].astype(str).str.lower().str.strip()
    
    try:
        df_calc['date_dt'] = pd.to_datetime(df_calc['date'], errors='coerce')
        df_calc = df_calc.dropna(subset=['date_dt'])
        df_calc['month'] = df_calc['date_dt'].dt.strftime('%Y-%m')
    except Exception:
        df_calc['month'] = df_calc['date'].astype(str).str.slice(0, 7)
        
    months = sorted(df_calc['month'].unique())
    rows = []
    
    for m in months:
        m_df = df_calc[df_calc['month'] == m]
        m_inc = round(float(m_df[m_df['type'] == 'credit']['amount'].sum()), 2)
        m_exp = round(float(m_df[m_df['type'] == 'debit']['amount'].sum()), 2)
        m_net = round(m_inc - m_exp, 2)
        rows.append({
            'month': m,
            'income': m_inc,
            'expenses': m_exp,
            'net_savings': m_net
        })
        
    return pd.DataFrame(rows)

def calculate_50_30_20_allocation(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluates budget adherence against the 50% Needs, 30% Wants, 20% Savings framework.
    Calculates overall Financial Health Score (0 - 100).
    """
    if df.empty:
        return {
            "needs": {"amount": 0.0, "actual_pct": 0.0, "target_pct": 50.0, "status": "No data"},
            "wants": {"amount": 0.0, "actual_pct": 0.0, "target_pct": 30.0, "status": "No data"},
            "savings": {"amount": 0.0, "actual_pct": 0.0, "target_pct": 20.0, "status": "No data"},
            "health_score": 50,
            "grade": "C",
            "assessment": "No transaction records found."
        }
        
    summary = calculate_summary_metrics(df)
    total_income = summary['total_income']
    
    df_calc = df.copy()
    if 'category' not in df_calc.columns:
        df_calc['category'] = 'Miscellaneous'
    df_calc['amount'] = pd.to_numeric(df_calc['amount'], errors='coerce').fillna(0.0)
    df_calc['type'] = df_calc['type'].astype(str).str.lower().str.strip()
    debits = df_calc[df_calc['type'] == 'debit']
    
    needs_total = 0.0
    wants_total = 0.0
    savings_explicit = 0.0
    
    for _, row in debits.iterrows():
        cat = str(row.get('category', 'Miscellaneous')).strip()
        amt = float(row.get('amount', 0.0))
        bucket = CATEGORY_BUDGET_MAP.get(cat, 'Wants')
        
        if bucket == 'Needs':
            needs_total += amt
        elif bucket == 'Savings':
            savings_explicit += amt
        else:
            wants_total += amt
            
    # If income was recorded, denominator is income; otherwise fallback to total expenditure
    base_denominator = total_income if total_income > 0 else (needs_total + wants_total + savings_explicit)
    if base_denominator <= 0:
        base_denominator = 1.0
        
    # Unspent cash flow also counts towards savings
    net_surplus = max(0.0, summary['net_cash_flow'])
    total_savings_effective = savings_explicit + (net_surplus if total_income > 0 else 0.0)
    
    needs_pct = round((needs_total / base_denominator) * 100, 1)
    wants_pct = round((wants_total / base_denominator) * 100, 1)
    savings_pct = round((total_savings_effective / base_denominator) * 100, 1)
    
    # Financial Health Scoring Algorithm (0 - 100)
    # 1. Savings rate component (up to 40 pts): 20%+ gives full 40 pts
    score_savings = min(40, (savings_pct / 20.0) * 40.0)
    
    # 2. Needs ratio component (up to 30 pts): <= 50% gives full 30 pts, drops if > 60%
    if needs_pct <= 50.0:
        score_needs = 30.0
    elif needs_pct <= 70.0:
        score_needs = max(0.0, 30.0 - ((needs_pct - 50.0) / 20.0) * 20.0)
    else:
        score_needs = 5.0
        
    # 3. Wants ratio component (up to 30 pts): <= 30% gives full 30 pts
    if wants_pct <= 30.0:
        score_wants = 30.0
    elif wants_pct <= 45.0:
        score_wants = max(0.0, 30.0 - ((wants_pct - 30.0) / 15.0) * 20.0)
    else:
        score_wants = 5.0
        
    health_score = int(round(score_savings + score_needs + score_wants))
    health_score = max(10, min(100, health_score))
    
    if health_score >= 85:
        grade = "A (Excellent)"
        assessment = "Superb financial discipline. Strong savings margin and tightly controlled discretionary spending."
    elif health_score >= 70:
        grade = "B (Healthy)"
        assessment = "Solid financial balance. Needs and wants are largely well managed with positive capital growth."
    elif health_score >= 55:
        grade = "C (Fair)"
        assessment = "Adequate, but higher discretionary wants or fixed obligations are constricting your savings capacity."
    else:
        grade = "D (Needs Attention)"
        assessment = "High burn rate and low savings cushion. Immediate expenditure optimization recommended."
        
    return {
        "needs": {
            "amount": round(needs_total, 2),
            "actual_pct": needs_pct,
            "target_pct": 50.0,
            "status": "Healthy" if needs_pct <= 50.0 else "Over Target"
        },
        "wants": {
            "amount": round(wants_total, 2),
            "actual_pct": wants_pct,
            "target_pct": 30.0,
            "status": "Healthy" if wants_pct <= 30.0 else "High Discretionary"
        },
        "savings": {
            "amount": round(total_savings_effective, 2),
            "actual_pct": savings_pct,
            "target_pct": 20.0,
            "status": "On Track" if savings_pct >= 20.0 else "Below 20% Goal"
        },
        "health_score": health_score,
        "grade": grade,
        "assessment": assessment
    }

def detect_recurring_expenses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies candidate recurring subscriptions or periodic bills based on repetitive descriptions.
    """
    if df.empty:
        return pd.DataFrame(columns=['description', 'category', 'frequency', 'avg_amount', 'total_paid'])
        
    df_calc = df.copy()
    if 'category' not in df_calc.columns:
        df_calc['category'] = 'Miscellaneous'
    df_calc['amount'] = pd.to_numeric(df_calc['amount'], errors='coerce').fillna(0.0)
    df_calc['type'] = df_calc['type'].astype(str).str.lower().str.strip()
    debits = df_calc[df_calc['type'] == 'debit']
    
    if debits.empty:
        return pd.DataFrame(columns=['description', 'category', 'frequency', 'avg_amount', 'total_paid'])
        
    grouped = debits.groupby('description').agg(
        category=('category', 'first'),
        frequency=('amount', 'count'),
        avg_amount=('amount', 'mean'),
        total_paid=('amount', 'sum')
    ).reset_index()
    
    # Filter for entries appearing >= 2 times or matching subscription keywords
    sub_keywords = ['netflix', 'spotify', 'prime', 'membership', 'gym', 'internet', 'verizon', 'con edison', 'rent']
    is_keyword_sub = grouped['description'].str.lower().apply(lambda x: any(kw in x for kw in sub_keywords))
    
    recurring = grouped[(grouped['frequency'] >= 2) | is_keyword_sub].copy()
    recurring['avg_amount'] = recurring['avg_amount'].round(2)
    recurring['total_paid'] = recurring['total_paid'].round(2)
    recurring = recurring.sort_values(by='total_paid', ascending=False).reset_index(drop=True)
    return recurring

def get_top_merchants(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Returns top debit merchants by aggregate spending.
    """
    if df.empty:
        return pd.DataFrame(columns=['merchant', 'total_spent', 'tx_count'])
        
    df_calc = df.copy()
    if 'category' not in df_calc.columns:
        df_calc['category'] = 'Miscellaneous'
    df_calc['amount'] = pd.to_numeric(df_calc['amount'], errors='coerce').fillna(0.0)
    df_calc['type'] = df_calc['type'].astype(str).str.lower().str.strip()
    debits = df_calc[df_calc['type'] == 'debit']
    
    if debits.empty:
        return pd.DataFrame(columns=['merchant', 'total_spent', 'tx_count'])
        
    top_merch = debits.groupby('description').agg(
        total_spent=('amount', 'sum'),
        tx_count=('amount', 'count')
    ).reset_index()
    
    top_merch.rename(columns={'description': 'merchant'}, inplace=True)
    top_merch['total_spent'] = top_merch['total_spent'].round(2)
    top_merch = top_merch.sort_values(by='total_spent', ascending=False).head(top_n).reset_index(drop=True)
    return top_merch
