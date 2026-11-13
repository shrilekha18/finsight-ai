"""
FinSight AI - Plotly Visualization Engine
Generates interactive, responsive financial charts for the Streamlit dashboard.
"""
from typing import Optional, Dict, Any
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def create_category_pie_chart(category_df: pd.DataFrame):
    """
    Creates an interactive donut chart showing expense allocation across categories.
    """
    if not PLOTLY_AVAILABLE or category_df.empty:
        return None
        
    fig = px.pie(
        category_df,
        names='category',
        values='total_amount',
        hole=0.45,
        title="<b>Expense Breakdown by Category</b>",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate="<b>%{label}</b><br>Spent: $%{value:,.2f}<br>Share: %{percent}<extra></extra>"
    )
    fig.update_layout(
        margin=dict(t=50, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        showlegend=True,
        template="plotly_white"
    )
    return fig


def create_monthly_income_expense_chart(trends_df: pd.DataFrame):
    """
    Creates a grouped bar chart comparing Monthly Income vs Expenses.
    """
    if not PLOTLY_AVAILABLE or trends_df.empty:
        return None
        
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=trends_df['month'],
        y=trends_df['income'],
        name='Income ($)',
        marker_color='#2ECC71',
        hovertemplate="Income: $%{y:,.2f}<extra></extra>"
    ))
    
    fig.add_trace(go.Bar(
        x=trends_df['month'],
        y=trends_df['expenses'],
        name='Expenses ($)',
        marker_color='#E74C3C',
        hovertemplate="Expenses: $%{y:,.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        barmode='group',
        title="<b>Monthly Income vs. Expenses</b>",
        xaxis_title="Billing Month",
        yaxis_title="Amount (USD $)",
        margin=dict(t=50, b=30, l=30, r=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white"
    )
    return fig


def create_cash_flow_trend_chart(trends_df: pd.DataFrame):
    """
    Creates a line chart with fill showing net cash flow trajectory across months.
    """
    if not PLOTLY_AVAILABLE or trends_df.empty:
        return None
        
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=trends_df['month'],
        y=trends_df['net_savings'],
        mode='lines+markers+text',
        name='Net Savings',
        text=[f"${val:,.0f}" for val in trends_df['net_savings']],
        textposition="top center",
        line=dict(color='#3498DB', width=3),
        marker=dict(size=8, symbol='circle'),
        fill='tozeroy',
        fillcolor='rgba(52, 152, 219, 0.1)',
        hovertemplate="Month: %{x}<br>Net Cash Flow: $%{y:,.2f}<extra></extra>"
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break Even")
    
    fig.update_layout(
        title="<b>Net Cash Flow Trajectory</b>",
        xaxis_title="Month",
        yaxis_title="Net Savings ($)",
        margin=dict(t=50, b=30, l=30, r=30),
        template="plotly_white"
    )
    return fig


def create_budget_benchmark_chart(budget_data: Dict[str, Any]):
    """
    Creates a comparative horizontal bar chart comparing Actual vs Target 50/30/20 proportions.
    """
    if not PLOTLY_AVAILABLE or not budget_data:
        return None
        
    categories = ['Needs', 'Wants', 'Savings']
    actual_pct = [
        budget_data.get('needs', {}).get('actual_pct', 0.0),
        budget_data.get('wants', {}).get('actual_pct', 0.0),
        budget_data.get('savings', {}).get('actual_pct', 0.0)
    ]
    target_pct = [50.0, 30.0, 20.0]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Actual % of Budget',
        y=categories,
        x=actual_pct,
        orientation='h',
        marker_color='#4A90E2',
        text=[f"{v:.1f}%" for v in actual_pct],
        textposition='auto',
        hovertemplate="Actual %: %{x:.1f}%<extra></extra>"
    ))
    
    fig.add_trace(go.Bar(
        name='50/30/20 Benchmark %',
        y=categories,
        x=target_pct,
        orientation='h',
        marker_color='#BDC3C7',
        text=[f"{v:.0f}%" for v in target_pct],
        textposition='auto',
        hovertemplate="Target %: %{x:.0f}%<extra></extra>"
    ))
    
    fig.update_layout(
        barmode='group',
        title="<b>50/30/20 Budget Rule Compliance</b>",
        xaxis_title="Percentage (%)",
        margin=dict(t=50, b=30, l=30, r=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white"
    )
    return fig


def create_top_merchants_chart(top_merchants_df: pd.DataFrame):
    """
    Creates a horizontal bar chart displaying top debit payees.
    """
    if not PLOTLY_AVAILABLE or top_merchants_df.empty:
        return None
        
    df_sorted = top_merchants_df.sort_values(by='total_spent', ascending=True)
    
    fig = px.bar(
        df_sorted,
        x='total_spent',
        y='merchant',
        orientation='h',
        title="<b>Top Spend Outlets</b>",
        labels={'total_spent': 'Total Outflow ($)', 'merchant': 'Merchant / Payee'},
        text='total_spent',
        color='total_spent',
        color_continuous_scale='Reds'
    )
    fig.update_traces(
        texttemplate='$%{text:,.2f}',
        textposition='outside',
        hovertemplate="Merchant: %{y}<br>Total: $%{x:,.2f}<extra></extra>"
    )
    fig.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(t=50, b=30, l=30, r=60),
        template="plotly_white"
    )
    return fig

