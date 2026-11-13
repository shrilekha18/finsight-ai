"""
FinSight AI - Multi-Agent Architecture Package
"""
from agents.parser_agent import ParserAgent
from agents.categorizer_agent import CategorizerAgent
from agents.analysis_agent import AnalysisAgent
from agents.rag_agent import RAGAgent
from agents.advisor_agent import AdvisorAgent
from agents.orchestrator_agent import OrchestratorAgent

__all__ = [
    "ParserAgent",
    "CategorizerAgent",
    "AnalysisAgent",
    "RAGAgent",
    "AdvisorAgent",
    "OrchestratorAgent"
]

