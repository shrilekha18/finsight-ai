"""
FinSight AI - Categorizer Agent
Employs LangChain Few-Shot Prompting, LCEL Chains, and Output Parsers
to intelligently categorize bank transactions into standard financial classifications.
"""
from typing import List, Dict, Any, Optional
import pandas as pd
from config import CATEGORIES
from agents.llm_factory import (
    get_llm,
    PromptTemplate,
    FewShotPromptTemplate,
    JsonOutputParser,
    StrOutputParser
)

class CategorizerAgent:
    """
    Categorizer Agent utilizing Few-Shot Prompting with LangChain LCEL chains.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.llm = get_llm(temperature=0.0, api_key=api_key)
        self.categories = CATEGORIES
        self._build_few_shot_chain()

        # Keyword heuristics for high-throughput batch classification and deterministic fallback
        self.rules = {
            "Income / Payroll": ["payroll", "direct deposit", "salary", "bonus", "dividend", "interest income", "stipend"],
            "Housing": ["rent", "mortgage", "property management", "apartment", "realty", "lease", "landlord", "hoa"],
            "Utilities": ["electric", "gas & electric", "water", "sewer", "con edison", "verizon", "at&t", "internet", "power", "utility"],
            "Groceries": ["supermarket", "whole foods", "trader joe", "safeway", "kroger", "costco", "aldi", "grocer", "market", "walmart grocery"],
            "Dining Out": ["restaurant", "bistro", "cafe", "starbucks", "chipotle", "diner", "pizza", "burger", "grill", "doordash", "ubereats", "bakery"],
            "Transportation": ["uber", "lyft", "shell oil", "chevron", "bp gas", "mobil", "transit", "subway", "metro", "parking", "toll", "exxon"],
            "Healthcare": ["pharmacy", "cvs", "walgreens", "hospital", "clinic", "dental", "doctor", "health", "urgent care", "labcorp", "quest"],
            "Entertainment": ["netflix", "spotify", "hulu", "disney", "cinema", "movie", "theater", "concert", "ticketmaster", "gym", "fitness", "equinox"],
            "Shopping": ["amazon", "target", "walmart", "best buy", "clothing", "apple store", "zara", "nike", "home depot", "retail"],
            "Investment & Savings": ["vanguard", "fidelity", "schwab", "robinhood", "etrade", "index fund", "etf", "coinbase", "transfer to savings"]
        }

    def _build_few_shot_chain(self):
        """
        Constructs a LangChain Few-Shot Prompt Template & LCEL chain.
        Demonstrates Few-Shot Prompt Engineering.
        """
        # Curated Few-Shot Examples
        examples = [
            {"description": "WHOLE FOODS MKT #1024", "amount": "114.20", "type": "debit", "category": "Groceries"},
            {"description": "NETFLIX.COM SUBSCRIPTION", "amount": "15.49", "type": "debit", "category": "Entertainment"},
            {"description": "APEX PROPERTY MGMT RENT", "amount": "1450.00", "type": "debit", "category": "Housing"},
            {"description": "CHEVRON GAS STATION", "amount": "45.00", "type": "debit", "category": "Transportation"},
            {"description": "ACME CORP DIRECT DEP PAYROLL", "amount": "3500.00", "type": "credit", "category": "Income / Payroll"},
            {"description": "CON EDISON ELEC & GAS", "amount": "88.50", "type": "debit", "category": "Utilities"},
            {"description": "CHIPOTLE ONLINE ORDER", "amount": "16.80", "type": "debit", "category": "Dining Out"},
            {"description": "VANGUARD TOTAL STOCK ETF", "amount": "500.00", "type": "debit", "category": "Investment & Savings"},
            {"description": "CVS PHARMACY RX", "amount": "28.50", "type": "debit", "category": "Healthcare"},
            {"description": "BEST BUY STORE #491", "amount": "89.99", "type": "debit", "category": "Shopping"}
        ]

        example_prompt = PromptTemplate(
            template="Transaction: {description} | Amount: ${amount} | Type: {type}\nCategory: {category}",
            input_variables=["description", "amount", "type", "category"]
        )

        prefix = (
            "You are an expert financial categorization assistant. "
            f"Classify the bank transaction into exactly one of these allowed categories:\n"
            f"{', '.join(self.categories)}.\n\n"
            "Here are verified reference examples:"
        )

        suffix = (
            "Now classify this transaction. Return JSON with 'category' and 'confidence' (0.0-1.0):\n"
            "Transaction: {description} | Amount: ${amount} | Type: {type}\n"
            "JSON Response:"
        )

        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=["description", "amount", "type"]
        )

        # LangChain LCEL Chain: Prompt -> LLM -> JSON Parser
        self.chain = few_shot_prompt | self.llm | JsonOutputParser()

    def categorize_single_llm(self, description: str, amount: float = 0.0, tx_type: str = "debit") -> str:
        """
        Invokes the LangChain Few-Shot chain to categorize a single transaction.
        """
        try:
            res = self.chain.invoke({
                "description": description,
                "amount": f"{amount:.2f}",
                "type": tx_type
            })
            cat = res.get("category", "").strip() if isinstance(res, dict) else str(res).strip()
            # Validate against allowed categories
            for valid_cat in self.categories:
                if valid_cat.lower() == cat.lower():
                    return valid_cat
            # Partial match
            for valid_cat in self.categories:
                if valid_cat.lower() in cat.lower():
                    return valid_cat
            return self.classify_by_rule(description, tx_type)
        except Exception:
            return self.classify_by_rule(description, tx_type)

    def classify_by_rule(self, description: str, tx_type: str = "debit") -> str:
        """
        Rule-based heuristic classifier ensuring rapid, zero-latency categorization.
        """
        desc_lower = str(description).lower()
        
        if tx_type.lower() == 'credit':
            if any(k in desc_lower for k in self.rules["Income / Payroll"]):
                return "Income / Payroll"
            if any(k in desc_lower for k in self.rules["Investment & Savings"]):
                return "Investment & Savings"
            return "Income / Payroll"
            
        for category, keywords in self.rules.items():
            if any(kw in desc_lower for kw in keywords):
                return category
                
        return "Miscellaneous"

    def categorize_dataframe(self, df: pd.DataFrame, use_llm_for_all: bool = False) -> pd.DataFrame:
        """
        Categorizes an entire DataFrame of transactions.
        Applies rules first for speed and accuracy; optionally invokes Few-Shot LLM for ambiguous items.
        """
        if df.empty:
            return df
            
        df_out = df.copy()
        categories = []
        
        for _, row in df_out.iterrows():
            desc = str(row.get('description', ''))
            amt = float(row.get('amount', 0.0))
            t_type = str(row.get('type', 'debit'))
            
            rule_cat = self.classify_by_rule(desc, t_type)
            if rule_cat != "Miscellaneous" and not use_llm_for_all:
                categories.append(rule_cat)
            else:
                # Use Few-Shot LLM chain
                llm_cat = self.categorize_single_llm(desc, amt, t_type)
                categories.append(llm_cat if llm_cat in self.categories else rule_cat)
                
        df_out['category'] = categories
        return df_out

