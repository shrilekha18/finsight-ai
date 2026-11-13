"""
FinSight AI - Personal Autonomous Financial Intelligence Agent
Main Streamlit Application
"""
import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# Add app directory to path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
import database
from agents.parser_agent import ParserAgent
from agents.categorizer_agent import CategorizerAgent
from agents.analysis_agent import AnalysisAgent
from agents.rag_agent import RAGAgent
from agents.advisor_agent import AdvisorAgent
from agents.orchestrator_agent import OrchestratorAgent
from utils.calculations import (
    calculate_summary_metrics,
    calculate_category_breakdown,
    calculate_monthly_trends,
    calculate_50_30_20_allocation,
    detect_recurring_expenses,
    get_top_merchants
)
from utils.visualizer import (
    create_category_pie_chart,
    create_monthly_income_expense_chart,
    create_cash_flow_trend_chart,
    create_budget_benchmark_chart,
    create_top_merchants_chart
)

# Page configuration
st.set_page_config(
    page_title="FinSight AI - Financial Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #4A90E2;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .intent-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        background-color: #E8F4FD;
        color: #1A73E8;
        margin-bottom: 8px;
    }
    .agent-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        background-color: #E6F4EA;
        color: #137333;
        margin-bottom: 8px;
        margin-left: 6px;
    }
    .stButton>button {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.title("💳 FinSight AI")
    st.caption("Autonomous Financial Intelligence Agent")
    st.divider()

    # Navigation
    nav_option = st.radio(
        "Navigation",
        ["📊 Dashboard", "📤 Upload Statement", "💬 AI Finance Chat", "💡 Financial Insights"],
        index=0
    )
    st.divider()

    # API Configuration
    st.subheader("⚙️ LLM Settings")
    api_key_input = st.text_input(
        "Gemini API Key",
        value=config.GEMINI_API_KEY,
        type="password",
        help="Enter your Google Gemini API key. If omitted, FinSight AI operates in resilient fallback mode."
    )
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input
        config.GEMINI_API_KEY = api_key_input
        st.success("API Key Active")
    else:
        st.info("ℹ️ Running in Local Mode")

    st.divider()
    st.subheader("🗂️ Quick Sample Data")
    if st.button("📥 Load Sample Statement", use_container_width=True):
        sample_path = BASE_DIR / "data" / "sample_statement.csv"
        if sample_path.exists():
            parser = ParserAgent()
            categorizer = CategorizerAgent(api_key=api_key_input)
            df, _ = parser.parse_statement(str(sample_path), "sample_statement.csv")
            df_cat = categorizer.categorize_dataframe(df)
            database.clear_all_transactions()
            inserted = database.insert_transactions(df_cat, source_file="sample_statement.csv")
            st.success(f"Loaded {inserted} sample transactions!")
            st.rerun()
        else:
            st.error("Sample CSV file not found.")

    if st.button("🗑️ Reset Database", use_container_width=True):
        database.clear_all_transactions()
        database.clear_chat_history()
        st.warning("Database reset.")
        st.rerun()

# Shared Agent Instances
parser_agent = ParserAgent()
categorizer_agent = CategorizerAgent(api_key=api_key_input)
analysis_agent = AnalysisAgent()
rag_agent = RAGAgent(api_key=api_key_input)
advisor_agent = AdvisorAgent(api_key=api_key_input)
orchestrator_agent = OrchestratorAgent(api_key=api_key_input)

# Fetch Current Transactions from SQLite
transactions_df = database.get_all_transactions()

# ----------------- 1. DASHBOARD -----------------
if nav_option == "📊 Dashboard":
    st.header("📊 Financial Intelligence Dashboard")
    st.caption("Real-time telemetry, cash flow analytics, and expenditure distributions")

    if transactions_df.empty:
        st.info("👋 Welcome to FinSight AI! No statement records loaded yet.")
        st.write("Click **'📥 Load Sample Statement'** in the sidebar to view demo analytics, or navigate to **'📤 Upload Statement'** to import your own CSV or PDF statement.")
    else:
        summary = calculate_summary_metrics(transactions_df)

        # Top Metric Cards
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Income", f"${summary['total_income']:,.2f}")
        with col2:
            st.metric("Total Expenses", f"${summary['total_expenses']:,.2f}")
        with col3:
            net_color = "normal" if summary['net_cash_flow'] >= 0 else "inverse"
            st.metric("Net Cash Flow", f"${summary['net_cash_flow']:,.2f}", delta=f"{summary['savings_rate']:.1f}% saved", delta_color=net_color)
        with col4:
            st.metric("Savings Rate", f"{summary['savings_rate']:.1f}%")
        with col5:
            st.metric("Daily Burn Rate", f"${summary['daily_burn_rate']:,.2f}/day")

        st.divider()

        # Visualizations Row 1
        cat_df = calculate_category_breakdown(transactions_df)
        top_merch_df = get_top_merchants(transactions_df, top_n=5)

        v_col1, v_col2 = st.columns(2)
        with v_col1:
            pie_fig = create_category_pie_chart(cat_df)
            if pie_fig:
                st.plotly_chart(pie_fig, use_container_width=True)
            else:
                st.write("No category data available.")
        with v_col2:
            merch_fig = create_top_merchants_chart(top_merch_df)
            if merch_fig:
                st.plotly_chart(merch_fig, use_container_width=True)
            else:
                st.write("No merchant spend data available.")

        # Visualizations Row 2
        trends_df = calculate_monthly_trends(transactions_df)
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            bar_fig = create_monthly_income_expense_chart(trends_df)
            if bar_fig:
                st.plotly_chart(bar_fig, use_container_width=True)
            else:
                st.write("Insufficient monthly data.")
        with t_col2:
            cf_fig = create_cash_flow_trend_chart(trends_df)
            if cf_fig:
                st.plotly_chart(cf_fig, use_container_width=True)
            else:
                st.write("Insufficient trend data.")

        # Recent Transactions
        st.subheader("📑 Recent Transactions")
        st.dataframe(
            transactions_df[['date', 'description', 'amount', 'type', 'category', 'balance']],
            use_container_width=True,
            hide_index=True
        )

# ----------------- 2. UPLOAD STATEMENT -----------------
elif nav_option == "📤 Upload Statement":
    st.header("📤 Statement Ingestion & Categorization")
    st.caption("Upload raw bank statements in CSV or PDF format for autonomous normalization and categorization.")

    uploaded_file = st.file_uploader(
        "Choose a bank statement (CSV or PDF)",
        type=["csv", "pdf"],
        help="Upload standard checking, credit card, or savings statements."
    )

    if uploaded_file is not None:
        file_name = uploaded_file.name
        st.success(f"Received file: **{file_name}** ({uploaded_file.size / 1024:.1f} KB)")

        col_run, col_mode = st.columns([2, 1])
        with col_mode:
            use_llm_few_shot = st.checkbox("Enable Few-Shot LLM for all rows", value=False, help="Uses LangChain Few-Shot prompting for every item (slower on large batches).")

        if st.button("🚀 Process & Ingest Statement", type="primary", use_container_width=True):
            with st.spinner("Parser Agent extracting transactions..."):
                parsed_df, parse_meta = parser_agent.parse_statement(uploaded_file, file_name)

            if not parse_meta['success'] or parsed_df.empty:
                st.error(f"Failed to parse statement: {parse_meta.get('error', 'No transactions found')}")
            else:
                st.info(f"Parser Agent extracted {len(parsed_df)} transactions using {parse_meta['parser_used']}.")

                with st.spinner("Categorizer Agent assigning financial classifications..."):
                    categorized_df = categorizer_agent.categorize_dataframe(parsed_df, use_llm_for_all=use_llm_few_shot)

                # Persist to SQLite
                inserted = database.insert_transactions(categorized_df, source_file=file_name)
                st.success(f"Successfully ingested {inserted} categorized transactions into SQLite database!")
                st.dataframe(categorized_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🗄️ Database Records Overview")
    if not transactions_df.empty:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            all_cats = ["All"] + sorted(list(transactions_df['category'].dropna().unique()))
            selected_cat = st.selectbox("Filter by Category", all_cats)
        with col_f2:
            selected_type = st.selectbox("Filter by Type", ["All", "Credit", "Debit"])

        filtered_df = database.get_filtered_transactions(
            category=selected_cat if selected_cat != "All" else None,
            tx_type=selected_type if selected_type != "All" else None
        )
        st.write(f"Displaying **{len(filtered_df)}** transactions:")
        st.dataframe(
            filtered_df[['date', 'description', 'amount', 'type', 'category', 'balance', 'source_file']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.write("No transactions saved in database.")

# ----------------- 3. AI FINANCE CHAT -----------------
elif nav_option == "💬 AI Finance Chat":
    st.header("💬 Autonomous Financial Chat")
    st.caption("Interact with the Orchestrator, Analysis, Advisor, and RAG Agents.")

    # Quick Prompts
    st.write("**Quick Prompts:**")
    prompt_cols = st.columns(4)
    quick_query = None
    if prompt_cols[0].button("🍽️ Dining Out Total", use_container_width=True):
        quick_query = "How much did I spend on dining out?"
    if prompt_cols[1].button("🚗 Find Uber Trips", use_container_width=True):
        quick_query = "Find all my Uber and transportation charges"
    if prompt_cols[2].button("⚖️ 50/30/20 Audit", use_container_width=True):
        quick_query = "How is my budget adhering to the 50/30/20 rule?"
    if prompt_cols[3].button("🏔️ Avalanche vs Snowball", use_container_width=True):
        quick_query = "What is the difference between debt avalanche and snowball?"

    # Conversation History from SQLite
    chat_history = database.get_chat_history(limit=30)
    for msg in chat_history:
        role = msg["role"]
        content = msg["content"]
        with st.chat_message(role):
            if role == "assistant" and msg.get("intent"):
                st.markdown(f'<span class="intent-badge">🎯 Intent: {msg["intent"]}</span>', unsafe_allow_html=True)
            st.markdown(content)

    # User Input
    user_input = st.chat_input("Ask any question about your spending, budget, or financial strategies...")
    active_prompt = quick_query or user_input

    if active_prompt:
        # Display user question
        with st.chat_message("user"):
            st.markdown(active_prompt)

        # Route through Orchestrator Agent
        with st.spinner("FinSight AI Orchestrator analyzing intent and coordinating agents..."):
            result = orchestrator_agent.route_and_execute(active_prompt, transactions_df)

        with st.chat_message("assistant"):
            # Metadata Badges
            intent_html = f'<span class="intent-badge">🎯 Intent: {result["intent"]} ({int(result["confidence"]*100)}%)</span>'
            agent_html = f'<span class="agent-badge">🤖 {result["active_agent"]}</span>'
            st.markdown(intent_html + agent_html, unsafe_allow_html=True)

            # Response
            st.markdown(result["response"])

            # ReAct Trace Expander (if Advisor Agent ran)
            if result.get("react_trace"):
                with st.expander("🔍 Autonomous ReAct Trace (Reasoning + Acting)"):
                    st.text(result["react_trace"])

            # Retrieved Knowledge Sources Expander (if RAG Agent ran)
            if result.get("retrieved_context"):
                with st.expander("📚 Retrieved Knowledge Context (ChromaDB)"):
                    for idx, doc in enumerate(result["retrieved_context"]):
                        st.markdown(f"**Source {idx+1}: {doc.get('source', 'Knowledge Base')}**")
                        st.caption(f"Relevance Score: {doc.get('similarity_score', 0.85)}")
                        st.markdown(f"> {doc.get('content')}")
                        st.divider()

    st.divider()
    if st.button("Clear Chat History"):
        database.clear_chat_history()
        st.rerun()

# ----------------- 4. FINANCIAL INSIGHTS -----------------
elif nav_option == "💡 Financial Insights":
    st.header("💡 Autonomous Financial Health Audit")
    st.caption("In-depth evaluation against standard wealth-building principles and budget targets.")

    if transactions_df.empty:
        st.warning("Please upload a bank statement or load sample data to generate financial insights.")
    else:
        budget_data = calculate_50_30_20_allocation(transactions_df)
        summary = calculate_summary_metrics(transactions_df)
        recurring_df = detect_recurring_expenses(transactions_df)

        # Health Score Card
        score = budget_data["health_score"]
        grade = budget_data["grade"]
        
        score_col1, score_col2 = st.columns([1, 2])
        with score_col1:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h3 style="margin: 0; color: #555;">Financial Health Score</h3>
                <h1 style="font-size: 3.5rem; margin: 10px 0; color: #1A73E8;">{score}<span style="font-size: 1.5rem; color: #888;">/100</span></h1>
                <h4 style="margin: 0; color: #2ECC71;">Grade: {grade}</h4>
            </div>
            """, unsafe_allow_html=True)
        with score_col2:
            st.subheader("Diagnostic Assessment")
            st.write(budget_data["assessment"])
            st.write(f"- **Monthly Net Margin:** ${summary['net_cash_flow']:,.2f}")
            st.write(f"- **Savings Rate:** {summary['savings_rate']:.1f}% of income")
            st.write(f"- **Burn Rate:** ${summary['daily_burn_rate']:,.2f} per active day")

        st.divider()

        # 50/30/20 Breakdown
        st.subheader("⚖️ 50/30/20 Budget Rule Allocation")
        b_col1, b_col2, b_col3 = st.columns(3)
        with b_col1:
            n = budget_data["needs"]
            st.metric("Needs (Target: 50%)", f"{n['actual_pct']:.1f}%", f"${n['amount']:,.2f}", delta_color="inverse" if n['actual_pct'] > 50 else "normal")
            st.caption(f"Status: **{n['status']}**")
        with b_col2:
            w = budget_data["wants"]
            st.metric("Wants (Target: 30%)", f"{w['actual_pct']:.1f}%", f"${w['amount']:,.2f}", delta_color="inverse" if w['actual_pct'] > 30 else "normal")
            st.caption(f"Status: **{w['status']}**")
        with b_col3:
            s = budget_data["savings"]
            st.metric("Savings (Target: 20%)", f"{s['actual_pct']:.1f}%", f"${s['amount']:,.2f}", delta_color="normal" if s['actual_pct'] >= 20 else "inverse")
            st.caption(f"Status: **{s['status']}**")

        bench_fig = create_budget_benchmark_chart(budget_data)
        if bench_fig:
            st.plotly_chart(bench_fig, use_container_width=True)

        st.divider()

        # Recurring Subscriptions Audit
        st.subheader("🔄 Detected Subscriptions & Recurring Bills")
        if not recurring_df.empty:
            st.write("Auditing recurring micro-charges is the fastest way to reduce fixed monthly burn:")
            st.dataframe(
                recurring_df[['description', 'category', 'frequency', 'avg_amount', 'total_paid']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No recurring subscription patterns detected.")

        st.divider()

        # Advisor Strategic Recommendations
        st.subheader("🤖 Autonomous Advisor Action Plan")
        if st.button("Generate Tailored Strategic Action Plan", type="primary"):
            with st.spinner("Advisor Agent executing ReAct cycle to generate action plan..."):
                adv_result = advisor_agent.run_react_cycle(
                    "Conduct a strategic review of my finances and deliver a 3-step action plan.",
                    transactions_df
                )
                st.markdown(adv_result["advice"])
                with st.expander("View ReAct Chain of Thought"):
                    st.text(adv_result["react_trace"])

