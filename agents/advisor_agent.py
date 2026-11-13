"""
FinSight AI - Advisor Agent
Executes a ReAct-style (Reasoning + Acting) autonomous agent workflow
to produce customized financial coaching, budgeting plans, and savings strategies.
"""
from typing import Dict, Any, List, Optional
import pandas as pd
from agents.llm_factory import get_llm, PromptTemplate, ChatPromptTemplate, StrOutputParser
from utils.calculations import (
    calculate_summary_metrics,
    calculate_category_breakdown,
    calculate_50_30_20_allocation,
    detect_recurring_expenses
)
from agents.rag_agent import RAGAgent

class AdvisorAgent:
    """
    Advisor Agent implements an explicit ReAct (Reason -> Act -> Observe -> Synthesize)
    agentic cycle to deliver personalized financial advisory recommendations.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.llm = get_llm(temperature=0.2, api_key=api_key)
        self.rag_agent = RAGAgent(api_key=api_key)
        self._build_react_chain()

    def _build_react_chain(self):
        """
        Builds the synthesis prompt template grounding ReAct observations into final advice.
        """
        template = (
            "You are FinSight AI's Lead Autonomous Financial Advisor.\n\n"
            "Here is the autonomous ReAct investigation trace regarding the user's financial status:\n"
            "--------------------------------------------------\n"
            "{react_trace}\n"
            "--------------------------------------------------\n\n"
            "USER'S INQUIRY OR FINANCIAL GOAL:\n"
            "{user_goal}\n\n"
            "ADVISORY INSTRUCTIONS:\n"
            "1. Synthesize the empirical observations into an empathetic, professional financial plan.\n"
            "2. Ground every recommendation in the user's actual metrics (quote exact dollar amounts and percentages).\n"
            "3. Provide 3 concrete, prioritized action steps the user can execute immediately.\n"
            "4. Do NOT hallucinate mathematical calculations; use the exact verified figures provided in the trace.\n\n"
            "AUTONOMOUS ADVISORY RECOMMENDATIONS:"
        )
        self.prompt = PromptTemplate.from_template(template)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def run_react_cycle(self, user_goal: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes an autonomous ReAct loop:
        - Step 1 (Thought): Reason what empirical calculations are required.
        - Step 2 (Action -> Observation): Compute summary metrics & 50/30/20 budget.
        - Step 3 (Thought): Reason whether knowledge base strategy retrieval is required.
        - Step 4 (Action -> Observation): Retrieve relevant financial principles via ChromaDB.
        - Step 5 (Final Answer): Synthesize comprehensive, personalized advice.
        """
        react_steps = []
        
        # --- TURN 1: Quantitative Audit ---
        thought_1 = "Thought 1: I need to inspect the user's factual financial cash flow, savings rate, and 50/30/20 budget distribution."
        react_steps.append(thought_1)
        
        # Action 1: Execute Python calculations
        action_1 = "Action 1: CalculateMetrics(transactions_df)"
        react_steps.append(action_1)
        
        summary = calculate_summary_metrics(df)
        budget = calculate_50_30_20_allocation(df)
        recurring = detect_recurring_expenses(df)
        
        observation_1 = (
            f"Observation 1: Total Inflows: ${summary['total_income']:,.2f} | "
            f"Total Outflows: ${summary['total_expenses']:,.2f} | "
            f"Net Cash Flow: ${summary['net_cash_flow']:,.2f} | "
            f"Savings Rate: {summary['savings_rate']:.1f}% | "
            f"Needs Allocation: {budget['needs']['actual_pct']:.1f}% (${budget['needs']['amount']:,.2f}) vs target 50% | "
            f"Wants Allocation: {budget['wants']['actual_pct']:.1f}% (${budget['wants']['amount']:,.2f}) vs target 30% | "
            f"Savings Allocation: {budget['savings']['actual_pct']:.1f}% vs target 20% | "
            f"Detected Subscriptions: {len(recurring)} active services."
        )
        react_steps.append(observation_1)
        
        # --- TURN 2: Knowledge Retrieval ---
        thought_2 = (
            "Thought 2: I must retrieve verified personal finance strategies from the knowledge base "
            "that address the user's specific financial profile and inquiry."
        )
        react_steps.append(thought_2)
        
        action_2 = f"Action 2: RAGKnowledgeRetrieval(query='{user_goal}')"
        react_steps.append(action_2)
        
        rag_res = self.rag_agent.generate_answer(user_goal)
        retrieved_snippets = []
        for d in rag_res.get("retrieved_context", []):
            retrieved_snippets.append(f"[{d['source']}]: {d['content'][:180]}...")
            
        observation_2 = (
            "Observation 2: Retrieved Financial Principles:\n" +
            ("\n".join(retrieved_snippets) if retrieved_snippets else "Standard financial rules applied.")
        )
        react_steps.append(observation_2)
        
        # --- TURN 3: Synthesis ---
        thought_3 = "Thought 3: Now I will synthesize the observations, calculate specific dollar targets, and generate an actionable financial plan."
        react_steps.append(thought_3)
        
        trace_str = "\n".join(react_steps)
        
        try:
            final_advice = self.chain.invoke({
                "react_trace": trace_str,
                "user_goal": user_goal
            })
        except Exception as e:
            final_advice = (
                f"### Strategic Financial Advisory\n\n"
                f"Based on your calculated cash flow:\n"
                f"- **Monthly Inflows:** ${summary['total_income']:,.2f}\n"
                f"- **Total Burn (Outflows):** ${summary['total_expenses']:,.2f}\n"
                f"- **Current Savings Margin:** ${summary['net_cash_flow']:,.2f} ({summary['savings_rate']:.1f}%)\n"
                f"- **50/30/20 Health Score:** {budget['health_score']}/100 ({budget['grade']})\n\n"
                f"**Priority Recommendations:**\n"
                f"1. **Emergency Reserve:** Build a 3-month living cushion (${budget['needs']['amount'] * 3:,.2f}) in a High-Yield Savings Account.\n"
                f"2. **Optimize Wants:** Your discretionary spend is currently ${budget['wants']['amount']:,.2f}. Shaving 10% redirects ${budget['wants']['amount'] * 0.1:,.2f} directly into investments.\n"
                f"3. **Automate Wealth:** Route ${summary['net_cash_flow'] * 0.5:,.2f} automatically on payday into broad-market index funds."
            )
            
        return {
            "advice": final_advice,
            "react_trace": trace_str,
            "metrics": summary,
            "budget": budget,
            "retrieved_sources": rag_res.get("retrieved_context", [])
        }

