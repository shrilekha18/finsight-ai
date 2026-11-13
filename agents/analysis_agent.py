"""
FinSight AI - Analysis Agent
Executes Python-driven deterministic calculations and financial aggregations.
Answers quantitative inquiries (spending totals, category breakdowns, monthly cash flows).
"""
from typing import Dict, Any, Optional
import pandas as pd
from utils.calculations import (
    calculate_summary_metrics,
    calculate_category_breakdown,
    calculate_monthly_trends,
    calculate_50_30_20_allocation,
    detect_recurring_expenses,
    get_top_merchants
)

class AnalysisAgent:
    """
    Analysis Agent delivers 100% accurate mathematical calculations
    for all financial queries, avoiding LLM hallucinations on numbers.
    """
    def __init__(self):
        pass

    def full_financial_audit(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs comprehensive quantitative analysis across all dimensions.
        """
        summary = calculate_summary_metrics(df)
        categories = calculate_category_breakdown(df)
        trends = calculate_monthly_trends(df)
        budget_503020 = calculate_50_30_20_allocation(df)
        recurring = detect_recurring_expenses(df)
        top_merchants = get_top_merchants(df, top_n=5)
        
        return {
            "summary": summary,
            "category_breakdown": categories.to_dict(orient="records"),
            "monthly_trends": trends.to_dict(orient="records"),
            "budget_allocation": budget_503020,
            "recurring_expenses": recurring.to_dict(orient="records"),
            "top_merchants": top_merchants.to_dict(orient="records")
        }

    def answer_spending_query(self, query: str, df: pd.DataFrame) -> str:
        """
        Interprets natural language spending questions and answers using Python calculations.
        """
        if df.empty:
            return "No transaction data is currently loaded. Please upload a bank statement first."
            
        q_lower = query.lower()
        summary = calculate_summary_metrics(df)
        categories_df = calculate_category_breakdown(df)
        
        # 1. Total spending question
        if any(k in q_lower for k in ["total spent", "total expenses", "how much did i spend in total", "overall spend"]):
            return (
                f"### Overall Spending Analysis\n"
                f"- **Total Inflows (Income):** ${summary['total_income']:,.2f}\n"
                f"- **Total Outflows (Expenses):** ${summary['total_expenses']:,.2f}\n"
                f"- **Net Savings:** ${summary['net_cash_flow']:,.2f} "
                f"({summary['savings_rate']:.1f}% savings rate across {summary['transaction_count']} transactions)."
            )
            
        # 2. Specific Category query
        for _, row in categories_df.iterrows():
            cat_name = str(row['category'])
            if cat_name.lower() in q_lower or (cat_name == "Dining Out" and "dining" in q_lower or "restaurants" in q_lower or "food" in q_lower):
                return (
                    f"### Category Spending: {cat_name}\n"
                    f"- **Total Spent:** ${row['total_amount']:,.2f}\n"
                    f"- **Transaction Count:** {int(row['transaction_count'])}\n"
                    f"- **Share of Total Expenses:** {row['percentage']:.1f}%\n"
                    f"- **Status:** Monitored under standard budget allocations."
                )
                
        # 3. Top spenders or largest expense
        if any(k in q_lower for k in ["biggest expense", "largest purchase", "highest charge", "most expensive"]):
            largest = summary['largest_expense']
            return (
                f"### Largest Single Expense\n"
                f"- **Merchant / Description:** {largest['description']}\n"
                f"- **Amount:** ${largest['amount']:,.2f}\n"
                f"- **Date Recorded:** {largest['date']}"
            )
            
        # 4. Recurring subscriptions
        if any(k in q_lower for k in ["subscription", "recurring", "monthly bill", "fixed cost"]):
            recurring_df = detect_recurring_expenses(df)
            if recurring_df.empty:
                return "No recurring subscription charges were detected in your statement."
            lines = [f"### Detected Recurring Bills & Subscriptions"]
            for _, r in recurring_df.head(6).iterrows():
                lines.append(f"- **{r['description']}**: ${r['avg_amount']:,.2f}/mo (Total: ${r['total_paid']:,.2f})")
            return "\n".join(lines)
            
        # 5. Fallback structured breakdown
        top_cats = categories_df.head(3)
        cat_lines = [f"- **{r['category']}**: ${r['total_amount']:,.2f} ({r['percentage']:.1f}%)" for _, r in top_cats.iterrows()]
        
        return (
            f"### Spending Summary\n"
            f"- **Total Outflows:** ${summary['total_expenses']:,.2f}\n"
            f"- **Net Cash Flow:** ${summary['net_cash_flow']:,.2f}\n"
            f"- **Top Spending Categories:**\n" + "\n".join(cat_lines)
        )

