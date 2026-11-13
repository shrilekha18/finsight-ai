# FinSight AI: Your Personal Autonomous Financial Intelligence Agent

FinSight AI is a production-grade, multi-agent personal financial intelligence application built with **Python**, **Streamlit**, **Google Gemini**, **LangChain**, **ChromaDB**, **Pandas**, **SQLite**, and **Plotly**.

It autonomously parses bank statements (PDF & CSV), normalizes transactions, classifies expenses using **Few-Shot prompt engineering**, detects user intent using **Zero-Shot prompt engineering**, retrieves domain financial wisdom via **Agentic RAG with conditional retrieval**, and formulates actionable, tailored financial coaching through an autonomous **ReAct-style reasoning agent**.

---

## 🏗️ Architecture & Multi-Agent System

```
                                  +-----------------------+
                                  |   Streamlit Web UI    |
                                  | (Dashboard, Ingestion,|
                                  |  Chat, Health Audit)  |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  |  Orchestrator Agent   |
                                  | (Zero-Shot Intent     |
                                  |  Detection & Router)  |
                                  +-----------+-----------+
                                              |
         +--------------------+---------------+---------------+--------------------+
         |                    |                               |                    |
         v                    v                               v                    v
+-----------------+  +-----------------+             +-----------------+  +-----------------+
|  Parser Agent   |  |Categorizer Agent|             | Analysis Agent  |  |   RAG Agent     |
| (CSV / PDF      |  | (Few-Shot LCEL  |             | (Pure Python &  |  | (ChromaDB +     |
|  Normalization) |  |  + Rules Engine)|             |  Pandas Math)   |  |  Cond. Retrieval|
+--------+--------+  +--------+--------+             +--------+--------+  +--------+--------+
         |                    |                               |                    |
         +----------+---------+                               |                    |
                    |                                         |                    |
                    v                                         v                    v
         +--------------------+                      +-------------------------------+
         |  SQLite Database   |                      |         Advisor Agent         |
         | (finsight.db)      |                      | (Autonomous ReAct Cycle:      |
         +--------------------+                      |  Thought -> Action -> Observe)|
                                                     +-------------------------------+
```

---

## 🌟 Mandatory Acceptance Objectives Implemented

| Objective | Implementation Details |
| :--- | :--- |
| **1. LLM Integration** | Google Gemini integration via LangChain `ChatGoogleGenerativeAI` with temperature tuning and resilient fallback support. |
| **2. Interactive GUI** | 4-section Streamlit application: **Dashboard**, **Upload Statement**, **AI Finance Chat**, and **Financial Insights**. |
| **3. Prompt Engineering** | **Zero-shot prompting** for intent routing; **Few-shot prompting** for categorization; **ReAct agent workflow** for advisory synthesis. |
| **4. LangChain Core** | `ChatPromptTemplate`, `PromptTemplate`, `FewShotPromptTemplate`, LCEL pipe syntax (`prompt \| llm \| parser`), `JsonOutputParser`, and `StrOutputParser`. |
| **5. Agentic RAG** | Financial knowledge base documents, ChromaDB vector database embeddings, **conditional retrieval gate**, and context-grounded answer generation. |
| **6. Intent Detection** | Zero-shot intent classifier routing between: `transaction_lookup`, `spending_analysis`, `budgeting_and_savings`, `financial_advice`, and `general_query`. |
| **7. Vector Database** | **ChromaDB** persistent storage for semantic similarity search with document metadata and distance metrics. |
| **8. Deterministic Math** | 100% of financial totals, ratios, burn rates, and 50/30/20 scores are calculated in **Python/Pandas** (no LLM math hallucinations). |

---

## 🤖 The 6 Autonomous Agents

1. **Parser Agent (`agents/parser_agent.py`)**
   - Ingests raw CSV and PDF statements.
   - Cleans messy headers, standardizes currency values, parses positive/negative signs into debits/credits, and formats dates to `YYYY-MM-DD`.
2. **Categorizer Agent (`agents/categorizer_agent.py`)**
   - Classifies transactions into standard categories (`Housing`, `Utilities`, `Groceries`, `Dining Out`, `Transportation`, `Healthcare`, `Entertainment`, `Shopping`, `Investment & Savings`, `Income / Payroll`, `Miscellaneous`).
   - Uses LangChain **Few-Shot prompt templates** with verified bank transaction examples + LCEL chain.
3. **Analysis Agent (`agents/analysis_agent.py`)**
   - Executes Python-driven mathematical aggregations: total income, total expenses, net cash flow, daily burn rate, top merchants, and category distributions.
4. **RAG Agent (`agents/rag_agent.py`)**
   - Employs **ChromaDB** with financial knowledge documents (`knowledge_base/`).
   - Features **Conditional Retrieval**: automatically determines whether a query is a direct calculation or requires conceptual financial wisdom before retrieving.
5. **Advisor Agent (`agents/advisor_agent.py`)**
   - Autonomous **ReAct Agent** executing sequential reasoning:
     - `Thought 1` -> `Action 1: CalculateMetrics()` -> `Observation 1`
     - `Thought 2` -> `Action 2: RAGKnowledgeRetrieval()` -> `Observation 2`
     - `Thought 3` -> `Synthesize Plan`
   - Formulates personalized 3-step financial improvement plans.
6. **Orchestrator Agent (`agents/orchestrator_agent.py`)**
   - Coordinates the multi-agent system.
   - Runs **Zero-Shot Intent Detection** to classify user questions and dynamically dispatch to the optimal agent.

---

## 🖥️ Streamlit GUI Sections

1. **📊 Dashboard**
   - Key Metric Cards: Total Income, Total Expenses, Net Cash Flow, Savings Rate (%), Daily Burn Rate.
   - Interactive Plotly Visualizations: Category Expense Donut Chart, Top 5 Payees Bar Chart, Monthly Income vs. Expenses Bar Chart, Net Cash Flow Trajectory.
   - Recent Transactions table.
2. **📤 Upload Statement**
   - Drag-and-drop file uploader for CSV and PDF bank statements.
   - Autonomous ingestion with real-time progress indicators.
   - SQLite persistent storage with category and type filtering.
3. **💬 AI Finance Chat**
   - Interactive chat interface with quick suggestion buttons.
   - Displays real-time **Intent Badges** (with confidence %) and **Active Agent Badges**.
   - Collapsible expanders showing the **Autonomous ReAct Reasoning Trace** and **Retrieved ChromaDB Knowledge Sources**.
4. **💡 Financial Insights**
   - **Autonomous Financial Health Audit**: Health Score (0-100) and Letter Grade (A/B/C/D).
   - **50/30/20 Rule Analysis**: Needs (50%), Wants (30%), Savings (20%) compliance visualizer.
   - **Subscription & Recurring Bill Detector**: Identifies recurring services (Netflix, Gym, Utilities, etc.) to target for cost reduction.
   - One-click tailored action plan generator.

---

## 📁 Project Directory Structure

```
finsight-ai/
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Python dependencies
├── .env.example                # Configuration template
├── .gitignore                  # Git ignore rules
├── README.md                   # System documentation
├── config.py                   # Central settings, paths, and budget targets
├── database.py                 # SQLite database layer (transactions & chat)
├── test_project.py             # Complete test suite (9 test modules)
├── data/
│   ├── sample_statement.csv    # Realistic bank statement (40 transactions)
│   ├── sample_statement.pdf    # Realistic PDF bank statement
│   └── generate_sample_pdf.py  # Script to generate PDF statement
├── knowledge_base/             # Curated financial knowledge documents for RAG
│   ├── 01_budgeting_frameworks.txt
│   ├── 02_emergency_funds.txt
│   ├── 03_debt_elimination_strategies.txt
│   ├── 04_investing_fundamentals.txt
│   └── 05_tax_and_savings_optimization.txt
├── utils/
│   ├── __init__.py
│   ├── calculations.py         # Pure Python deterministic math engine
│   ├── visualizer.py           # Plotly interactive charting utilities
│   └── pdf_parser.py           # PDF statement text & transaction extraction
└── agents/
    ├── __init__.py             # Agent exports
    ├── llm_factory.py          # Centralized LLM factory with LangChain & fallback
    ├── parser_agent.py         # Parser Agent (CSV & PDF)
    ├── categorizer_agent.py    # Categorizer Agent (Few-Shot LCEL)
    ├── analysis_agent.py       # Analysis Agent (Python calculations)
    ├── rag_agent.py            # Agentic RAG Agent (ChromaDB + Conditional Retrieval)
    ├── advisor_agent.py        # Advisor Agent (ReAct Reasoning Cycle)
    └── orchestrator_agent.py   # Orchestrator Agent (Zero-Shot Intent Detection)
```

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
- Python 3.9, 3.10, or 3.11 installed.

### 2. Setup Virtual Environment
```bash
# Clone or navigate to finsight-ai
cd finsight-ai

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Key
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and add your Google Gemini API key:
```env
GEMINI_API_KEY=AIzaSy...your_gemini_api_key_here
```
*(Note: You can also enter or update your Gemini API key directly in the Streamlit sidebar at any time).*

### 5. Run Verification Tests
Verify that all 17 modules, database tables, and multi-agent workflows pass:
```bash
python test_project.py
```

### 6. Launch the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Testing the Live Application

1. Click **"📥 Load Sample Statement"** in the sidebar to populate the database with realistic sample transactions.
2. Explore the **📊 Dashboard** to view income, expenses, savings rate, and Plotly charts.
3. Switch to **💬 AI Finance Chat** and ask:
   - *"How much did I spend on dining out?"* -> Handled by **Analysis Agent** (computes exact math).
   - *"Find all my Uber transactions"* -> Handled by **Parser & Database Retrieval**.
   - *"How is my 50/30/20 budget doing?"* -> Handled by **Advisor Agent** (triggers ReAct cycle).
   - *"What is the difference between debt avalanche and snowball?"* -> Handled by **RAG Agent** (retrieves verified knowledge).
4. Go to **💡 Financial Insights** to review your Financial Health Score (0-100), detected subscriptions, and click **"Generate Tailored Strategic Action Plan"**.
5. Upload your own `.csv` or `.pdf` statement in **📤 Upload Statement**!

---

## 🔒 Security & Privacy
- All financial calculations execute **locally on your machine**.
- SQLite database (`data/finsight.db`) and ChromaDB vector store (`data/chroma_db`) remain entirely private and local.
- No transaction details are sent to external services except the specific prompts dispatched to the LLM.

