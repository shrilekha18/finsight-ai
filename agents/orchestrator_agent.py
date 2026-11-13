"""
FinSight AI - Orchestrator Agent
Central coordinator and intent router.
Employs Zero-Shot Prompting with LangChain ChatPromptTemplate and LCEL chains
to classify user intent and dynamically dispatch tasks across specialized agents.
"""
from typing import Dict, Any, List, Optional
import json
import pandas as pd
from agents.llm_factory import (
    get_llm,
    ChatPromptTemplate,
    PromptTemplate,
    JsonOutputParser,
    StrOutputParser
)
from agents.analysis_agent import AnalysisAgent
from agents.rag_agent import RAGAgent
from agents.advisor_agent import AdvisorAgent
import database

class OrchestratorAgent:
    """
    Orchestrator Agent coordinates multi-agent workflows:
    - Zero-Shot Intent Detection (transaction_lookup, spending_analysis, budgeting_and_savings, financial_advice, general_query)
    - Dynamic Routing to Analysis, RAG, Advisor, or Database Lookup
    - Conversation State & Response Synthesis
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.llm = get_llm(temperature=0.0, api_key=api_key)
        self.analysis_agent = AnalysisAgent()
        self.rag_agent = RAGAgent(api_key=api_key)
        self.advisor_agent = AdvisorAgent(api_key=api_key)
        self._build_intent_chain()

    def _build_intent_chain(self):
        """
        Builds Zero-Shot Prompt Template with LangChain LCEL for Intent Classification.
        Demonstrates Zero-Shot Prompt Engineering.
        """
        system_prompt = (
            "You are an expert autonomous financial query router. "
            "Analyze the user's inquiry and classify its primary intent into exactly ONE of the following 5 categories:\n\n"
            "1. transaction_lookup: Searching for specific transactions, payees, dates, receipts, or record listings.\n"
            "2. spending_analysis: Asking about quantitative amounts spent, category breakdowns, highest expenses, or burn rate.\n"
            "3. budgeting_and_savings: Asking about budget frameworks (50/30/20), emergency fund rules, savings percentages, or expense trimming.\n"
            "4. financial_advice: Seeking strategic coaching, investment advice, debt elimination (avalanche/snowball), or a comprehensive plan.\n"
            "5. general_query: General questions, greetings, or conceptual definitions not tied to personal numbers.\n\n"
            "Respond ONLY with a valid JSON object with keys:\n"
            "{\n"
            '  "intent": "<one of the 5 exact category strings above>",\n'
            '  "confidence": <float between 0.0 and 1.0>,\n'
            '  "reasoning": "<brief 1-sentence justification>"\n'
            "}"
        )
        user_prompt = "User Inquiry: {query}"

        intent_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_prompt)
        ])

        # LangChain LCEL: ChatPromptTemplate | LLM | JsonOutputParser
        self.intent_chain = intent_prompt | self.llm | JsonOutputParser()

    def detect_intent(self, query: str) -> Dict[str, Any]:
        """
        Executes Zero-Shot Intent Classification.
        Includes robust keyword fallback for offline reliability.
        """
        try:
            result = self.intent_chain.invoke({"query": query})
            if isinstance(result, dict) and "intent" in result:
                intent = result["intent"].lower().strip()
                valid_intents = [
                    "transaction_lookup",
                    "spending_analysis",
                    "budgeting_and_savings",
                    "financial_advice",
                    "general_query"
                ]
                if intent in valid_intents:
                    return result
        except Exception:
            pass

        # Heuristic zero-shot fallback
        q_low = query.lower()
        if any(k in q_low for k in ["find", "search", "lookup", "when did", "uber", "netflix", "show me transactions", "receipt", "starbucks"]):
            return {"intent": "transaction_lookup", "confidence": 0.90, "reasoning": "Keyword match for specific transaction search."}
        elif any(k in q_low for k in ["how much", "total spent", "spent on", "groceries", "dining", "outflows", "biggest expense", "cost"]):
            return {"intent": "spending_analysis", "confidence": 0.92, "reasoning": "Keyword match for quantitative spending audit."}
        elif any(k in q_low for k in ["50/30/20", "budget", "emergency fund", "save", "savings rate", "cut costs", "burn rate"]):
            return {"intent": "budgeting_and_savings", "confidence": 0.94, "reasoning": "Keyword match for budget frameworks and savings rules."}
        elif any(k in q_low for k in ["advice", "invest", "debt", "avalanche", "snowball", "recommend", "plan", "strategy"]):
            return {"intent": "financial_advice", "confidence": 0.91, "reasoning": "Keyword match for financial coaching or strategy."}
        else:
            return {"intent": "general_query", "confidence": 0.85, "reasoning": "General informational inquiry."}

    def execute_transaction_lookup(self, query: str, df: pd.DataFrame) -> str:
        """
        Performs targeted lookup in transactions DataFrame/SQLite.
        """
        if df.empty:
            return "No transactions are currently loaded. Please upload a bank statement first."
            
        q_words = [w.strip() for w in query.lower().split() if len(w.strip()) > 2 and w not in ['the', 'and', 'for', 'show', 'find', 'transactions', 'when', 'did', 'pay', 'much']]
        
        matches = pd.DataFrame()
        for w in q_words:
            matched_rows = df[df['description'].str.lower().str.contains(w, regex=False, na=False)]
            if not matched_rows.empty:
                matches = pd.concat([matches, matched_rows]).drop_duplicates()
                
        if matches.empty:
            return f"No transactions matching your query keywords were found in the statement."
            
        lines = [f"### Found {len(matches)} Matching Transaction(s):"]
        for _, row in matches.head(8).iterrows():
            lines.append(f"- **{row['date']}** | {row['description']} | **${float(row['amount']):,.2f}** ({row['type'].upper()}) [{row.get('category', 'Uncategorized')}]")
            
        if len(matches) > 8:
            lines.append(f"*...and {len(matches) - 8} more transactions.*")
            
        return "\n".join(lines)

    def route_and_execute(self, query: str, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Dispatches query to appropriate specialized agent based on detected intent:
        - transaction_lookup -> Direct Database / DF search
        - spending_analysis -> Analysis Agent
        - budgeting_and_savings -> Advisor Agent / RAG Agent
        - financial_advice -> Advisor Agent (ReAct cycle)
        - general_query -> RAG Agent (ChromaDB retrieval)
        """
        if df is None:
            df = database.get_all_transactions()
            
        intent_data = self.detect_intent(query)
        intent = intent_data.get("intent", "general_query")
        confidence = intent_data.get("confidence", 0.90)
        
        retrieved_context = []
        react_trace = None
        
        if intent == "transaction_lookup":
            response_text = self.execute_transaction_lookup(query, df)
            active_agent = "Parser & Database Retrieval Agent"
            
        elif intent == "spending_analysis":
            response_text = self.analysis_agent.answer_spending_query(query, df)
            active_agent = "Analysis Agent (Deterministic Python Math)"
            
        elif intent == "budgeting_and_savings":
            if not df.empty:
                res = self.advisor_agent.run_react_cycle(query, df)
                response_text = res["advice"]
                react_trace = res.get("react_trace")
                retrieved_context = res.get("retrieved_sources", [])
                active_agent = "Advisor Agent (ReAct Budgeting)"
            else:
                rag_res = self.rag_agent.generate_answer(query)
                response_text = rag_res["answer"]
                retrieved_context = rag_res.get("retrieved_context", [])
                active_agent = "RAG Agent (Financial Knowledge Base)"
                
        elif intent == "financial_advice":
            if not df.empty:
                res = self.advisor_agent.run_react_cycle(query, df)
                response_text = res["advice"]
                react_trace = res.get("react_trace")
                retrieved_context = res.get("retrieved_sources", [])
                active_agent = "Advisor Agent (ReAct Planning Loop)"
            else:
                rag_res = self.rag_agent.generate_answer(query)
                response_text = rag_res["answer"]
                retrieved_context = rag_res.get("retrieved_context", [])
                active_agent = "RAG Agent (ChromaDB Guidance)"
                
        else: # general_query
            rag_res = self.rag_agent.generate_answer(query)
            response_text = rag_res["answer"]
            retrieved_context = rag_res.get("retrieved_context", [])
            active_agent = "RAG Agent (Semantic Knowledge Engine)"
            
        # Record turn in SQLite database
        database.save_chat_message(role="user", content=query, intent=intent)
        database.save_chat_message(role="assistant", content=response_text, intent=intent)
        
        return {
            "query": query,
            "intent": intent,
            "confidence": confidence,
            "reasoning": intent_data.get("reasoning", ""),
            "active_agent": active_agent,
            "response": response_text,
            "react_trace": react_trace,
            "retrieved_context": retrieved_context
        }

