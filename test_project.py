"""
FinSight AI - Comprehensive Verification & Test Suite
Validates syntax, database operations, calculation accuracy, and all multi-agent workflows.
"""
import os
import sys
import py_compile
import unittest
from pathlib import Path
import pandas as pd

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

class TestFinSightAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n" + "="*60)
        print("STARTING FINSIGHT AI SYSTEM VERIFICATION")
        print("="*60)

    def test_01_syntax_compilation(self):
        """Test that all python files in the project compile without syntax errors."""
        print("\n[TEST 1] Verifying Python syntax compilation...")
        python_files = list(PROJECT_ROOT.rglob("*.py"))
        for py_file in python_files:
            rel_path = py_file.relative_to(PROJECT_ROOT)
            try:
                py_compile.compile(str(py_file), doraise=True)
                print(f"  OK: {rel_path}")
            except py_compile.PyCompileError as e:
                self.fail(f"Syntax error in {rel_path}: {e}")

    def test_02_database_operations(self):
        """Test SQLite database operations and schema integrity."""
        print("\n[TEST 2] Testing SQLite database operations...")
        import database
        database.init_db()
        database.clear_all_transactions()
        database.clear_chat_history()

        # Insert test records
        test_df = pd.DataFrame([
            {"date": "2024-01-01", "description": "Payroll Direct Deposit", "amount": 3000.0, "type": "credit", "category": "Income / Payroll", "balance": 3000.0},
            {"date": "2024-01-02", "description": "Rental Office", "amount": 1200.0, "type": "debit", "category": "Housing", "balance": 1800.0},
            {"date": "2024-01-05", "description": "Grocery Store", "amount": 150.0, "type": "debit", "category": "Groceries", "balance": 1650.0}
        ])
        inserted = database.insert_transactions(test_df, source_file="test_suite")
        self.assertEqual(inserted, 3)

        # Retrieve records
        records = database.get_all_transactions()
        self.assertEqual(len(records), 3)

        # Filtered query
        groceries = database.get_filtered_transactions(category="Groceries")
        self.assertEqual(len(groceries), 1)

        # Chat history
        database.save_chat_message(role="user", content="Hello test", intent="general_query")
        chat = database.get_chat_history()
        self.assertTrue(len(chat) >= 1)
        print("  OK: Database operations verified.")

    def test_03_financial_calculations(self):
        """Test Python calculations for financial totals and ratios."""
        print("\n[TEST 3] Testing deterministic Python calculations...")
        from utils import calculations
        
        test_df = pd.DataFrame([
            {"date": "2024-01-01", "description": "Salary", "amount": 5000.0, "type": "credit", "category": "Income / Payroll"},
            {"date": "2024-01-02", "description": "Rent", "amount": 1500.0, "type": "debit", "category": "Housing"},
            {"date": "2024-01-03", "description": "Food Market", "amount": 500.0, "type": "debit", "category": "Groceries"},
            {"date": "2024-01-04", "description": "Fine Dining", "amount": 200.0, "type": "debit", "category": "Dining Out"},
            {"date": "2024-01-05", "description": "Netflix", "amount": 15.0, "type": "debit", "category": "Entertainment"}
        ])

        summary = calculations.calculate_summary_metrics(test_df)
        self.assertEqual(summary['total_income'], 5000.0)
        self.assertEqual(summary['total_expenses'], 2215.0)
        self.assertEqual(summary['net_cash_flow'], 2785.0)
        self.assertAlmostEqual(summary['savings_rate'], 55.7, places=1)

        budget = calculations.calculate_50_30_20_allocation(test_df)
        self.assertIn("health_score", budget)
        self.assertTrue(0 <= budget["health_score"] <= 100)
        print(f"  OK: Calculations verified. Health score: {budget['health_score']}")

    def test_04_parser_agent(self):
        """Test Parser Agent normalization on CSV statement."""
        print("\n[TEST 4] Testing Parser Agent on sample CSV...")
        from agents.parser_agent import ParserAgent
        parser = ParserAgent()
        sample_csv = PROJECT_ROOT / "data" / "sample_statement.csv"
        self.assertTrue(sample_csv.exists(), "Sample CSV must exist.")

        df, meta = parser.parse_statement(str(sample_csv), "sample_statement.csv")
        self.assertTrue(meta["success"])
        self.assertEqual(len(df), 40)
        self.assertListEqual(list(df.columns), ['date', 'description', 'amount', 'type', 'balance'])
        print(f"  OK: Parser Agent parsed {len(df)} transactions cleanly.")

    def test_05_categorizer_agent(self):
        """Test Categorizer Agent few-shot setup and rule categorization."""
        print("\n[TEST 5] Testing Categorizer Agent...")
        from agents.categorizer_agent import CategorizerAgent
        categorizer = CategorizerAgent()
        
        # Test rule classification
        cat_groceries = categorizer.classify_by_rule("Trader Joe's Supermarket", "debit")
        self.assertEqual(cat_groceries, "Groceries")
        
        cat_rent = categorizer.classify_by_rule("Apex Property Rent", "debit")
        self.assertEqual(cat_rent, "Housing")

        cat_salary = categorizer.classify_by_rule("Acme Payroll Direct Deposit", "credit")
        self.assertEqual(cat_salary, "Income / Payroll")
        print("  OK: Categorizer Agent rules verified.")

    def test_06_analysis_agent(self):
        """Test Analysis Agent query evaluation."""
        print("\n[TEST 6] Testing Analysis Agent...")
        from agents.analysis_agent import AnalysisAgent
        from agents.parser_agent import ParserAgent
        from agents.categorizer_agent import CategorizerAgent
        
        parser = ParserAgent()
        categorizer = CategorizerAgent()
        df, _ = parser.parse_statement(str(PROJECT_ROOT / "data" / "sample_statement.csv"))
        df_cat = categorizer.categorize_dataframe(df)

        analyzer = AnalysisAgent()
        res = analyzer.answer_spending_query("How much did I spend on groceries?", df_cat)
        self.assertIn("Groceries", res)
        self.assertIn("$", res)
        print(f"  OK: Analysis Agent answered spending query.")

    def test_07_rag_agent(self):
        """Test Agentic RAG conditional retrieval and knowledge lookup."""
        print("\n[TEST 7] Testing Agentic RAG Agent...")
        from agents.rag_agent import RAGAgent
        rag = RAGAgent()
        
        # Test conditional retrieval gate
        should_ret, _ = rag.should_retrieve("What is the 50/30/20 budgeting rule?")
        self.assertTrue(should_ret)

        should_not_ret, _ = rag.should_retrieve("how much did i spend on groceries")
        self.assertFalse(should_not_ret)

        # Test knowledge search
        results = rag.search_knowledge("emergency fund", top_k=2)
        self.assertTrue(len(results) > 0)
        self.assertIn("content", results[0])
        print(f"  OK: RAG Agent retrieved {len(results)} context chunks from knowledge base.")

    def test_08_advisor_agent_react(self):
        """Test Advisor Agent ReAct workflow."""
        print("\n[TEST 8] Testing Advisor Agent ReAct workflow...")
        from agents.advisor_agent import AdvisorAgent
        from agents.parser_agent import ParserAgent
        from agents.categorizer_agent import CategorizerAgent

        parser = ParserAgent()
        categorizer = CategorizerAgent()
        df, _ = parser.parse_statement(str(PROJECT_ROOT / "data" / "sample_statement.csv"))
        df_cat = categorizer.categorize_dataframe(df)

        advisor = AdvisorAgent()
        res = advisor.run_react_cycle("Help me save more money", df_cat)
        self.assertIn("react_trace", res)
        self.assertIn("Thought 1", res["react_trace"])
        self.assertIn("Action 1", res["react_trace"])
        self.assertIn("Observation 1", res["react_trace"])
        self.assertIn("advice", res)
        print("  OK: Advisor Agent ReAct loop executed successfully.")

    def test_09_orchestrator_agent(self):
        """Test Orchestrator Agent Zero-Shot Intent Detection & Routing."""
        print("\n[TEST 9] Testing Orchestrator Agent intent detection...")
        from agents.orchestrator_agent import OrchestratorAgent
        orchestrator = OrchestratorAgent()

        # Intent: spending_analysis
        intent_spend = orchestrator.detect_intent("How much did I spend on dining out?")
        self.assertEqual(intent_spend["intent"], "spending_analysis")

        # Intent: transaction_lookup
        intent_lookup = orchestrator.detect_intent("Find my Uber receipts")
        self.assertEqual(intent_lookup["intent"], "transaction_lookup")

        # Intent: budgeting_and_savings
        intent_budget = orchestrator.detect_intent("Explain the 50/30/20 budget framework")
        self.assertEqual(intent_budget["intent"], "budgeting_and_savings")

        # Intent: financial_advice
        intent_advice = orchestrator.detect_intent("Should I use debt avalanche or debt snowball?")
        self.assertEqual(intent_advice["intent"], "financial_advice")

        print("  OK: Orchestrator Zero-Shot Intent Detection fully verified across all categories.")

if __name__ == "__main__":
    unittest.main()

