"""
FinSight AI - Agentic RAG (Retrieval-Augmented Generation) Agent
Manages ChromaDB vector store, document chunking, embedding generation,
conditional retrieval, and grounded financial answer generation.
"""
import os
import glob
from typing import List, Dict, Any, Optional, Tuple
from config import KNOWLEDGE_BASE_DIR, CHROMA_PERSIST_DIR
from agents.llm_factory import get_llm, PromptTemplate, ChatPromptTemplate, StrOutputParser

# Check ChromaDB availability
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


class RAGAgent:
    """
    Agentic RAG Agent capable of:
    1. Ingesting financial knowledge documents into ChromaDB.
    2. Conditional retrieval (deciding when retrieval is required).
    3. Semantic similarity search.
    4. Synthesizing answers strictly grounded in retrieved evidence.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.llm = get_llm(temperature=0.1, api_key=api_key)
        self.collection_name = "finsight_financial_knowledge"
        self.chroma_client = None
        self.collection = None
        self.fallback_docs: List[Dict[str, str]] = []
        
        self._init_vector_store()
        self._build_synthesis_chain()

    def _init_vector_store(self):
        """
        Initializes ChromaDB persistent client and indexes knowledge base documents.
        """
        # Load raw knowledge documents
        doc_files = glob.glob(str(KNOWLEDGE_BASE_DIR / "*.txt"))
        raw_chunks = []
        
        for fpath in doc_files:
            fname = os.path.basename(fpath)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                # Chunk document by double newlines (paragraphs/sections)
                paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 30]
                for idx, p in enumerate(paragraphs):
                    raw_chunks.append({
                        "id": f"{fname}_chunk_{idx}",
                        "source": fname,
                        "text": p
                    })
            except Exception as e:
                print(f"Error loading doc {fpath}: {e}")
                
        self.fallback_docs = raw_chunks

        if CHROMADB_AVAILABLE:
            try:
                self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
                self.collection = self.chroma_client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"description": "Financial Wisdom and Strategy Guides"}
                )
                
                # Check if already populated
                existing_count = self.collection.count()
                if existing_count < len(raw_chunks):
                    ids = [c["id"] for c in raw_chunks]
                    documents = [c["text"] for c in raw_chunks]
                    metadatas = [{"source": c["source"]} for c in raw_chunks]
                    self.collection.upsert(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas
                    )
            except Exception as e:
                print(f"Warning: ChromaDB initialization error: {e}. Using resilient in-memory retriever.")
                self.chroma_client = None
                self.collection = None

    def should_retrieve(self, query: str) -> Tuple[bool, str]:
        """
        Agentic Conditional Retrieval Gate.
        Evaluates whether a user query requires financial domain knowledge retrieval,
        or if it is purely transactional/conversational.
        """
        q_lower = query.lower()
        
        # Pure transaction lookup/math queries DO NOT require conceptual RAG retrieval
        non_retrieval_triggers = [
            "how much did i spend",
            "show transactions",
            "what is my total",
            "who did i pay",
            "list expenses",
            "balance on",
            "search statement"
        ]
        if any(trigger in q_lower for trigger in non_retrieval_triggers) and not any(k in q_lower for k in ["rule", "strategy", "method", "guide", "framework"]):
            return False, "Query is transactional calculation; conditional retrieval bypassed."

        # Concept & Strategy triggers
        knowledge_triggers = [
            "50/30/20", "avalanche", "snowball", "emergency fund",
            "invest", "debt", "budget", "retirement", "hysa", "index fund",
            "savings rate", "401k", "ira", "tax", "sinking fund", "pay yourself first"
        ]
        if any(k in q_lower for k in knowledge_triggers):
            return True, "Financial concept detected; executing ChromaDB semantic retrieval."
            
        return True, "General inquiry; conditional retrieval active."

    def search_knowledge(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs semantic similarity search against the ChromaDB vector store.
        """
        if self.collection and self.collection.count() > 0:
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=min(top_k, self.collection.count())
                )
                retrieved = []
                docs = results.get("documents", [[]])[0]
                metas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [[]])[0] if "distances" in results else [0.0] * len(docs)
                
                for doc_text, meta, dist in zip(docs, metas, distances):
                    retrieved.append({
                        "content": doc_text,
                        "source": meta.get("source", "Knowledge Base"),
                        "similarity_score": round(1.0 - (dist if dist is not None else 0.0), 3)
                    })
                return retrieved
            except Exception as e:
                print(f"ChromaDB search error: {e}. Falling back to keyword search.")

        # Fallback In-Memory Retrieval
        q_words = set(query.lower().split())
        scored = []
        for item in self.fallback_docs:
            doc_words = set(item["text"].lower().split())
            overlap = len(q_words.intersection(doc_words))
            scored.append((overlap, item))
            
        scored.sort(key=lambda x: x[0], reverse=True)
        top_matches = [x[1] for x in scored[:top_k]]
        return [{
            "content": m["text"],
            "source": m["source"],
            "similarity_score": 0.85
        } for m in top_matches]

    def _build_synthesis_chain(self):
        """
        Constructs the LangChain LCEL chain for context-grounded response generation.
        """
        template = (
            "You are FinSight AI's Financial Intelligence Specialist.\n"
            "Answer the user's financial question strictly using the provided verified reference context.\n\n"
            "CONTEXT FROM VERIFIED FINANCIAL KNOWLEDGE BASE:\n"
            "{context}\n\n"
            "USER QUESTION:\n"
            "{question}\n\n"
            "INSTRUCTIONS:\n"
            "1. Ground your recommendations directly in the provided context.\n"
            "2. Be concise, actionable, and structured with bullet points.\n"
            "3. Cite the relevant strategy or principle mentioned in the context.\n"
            "4. Do not speculate or invent facts not present in the context.\n\n"
            "FINANCIAL GUIDANCE:"
        )
        self.prompt = PromptTemplate.from_template(template)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate_answer(self, query: str) -> Dict[str, Any]:
        """
        Executes the complete Agentic RAG cycle:
        1. Evaluates conditional retrieval gate.
        2. Retrieves relevant chunks if condition met.
        3. Synthesizes context-grounded response using LangChain LCEL chain.
        """
        needs_retrieval, reason = self.should_retrieve(query)
        
        if not needs_retrieval:
            return {
                "answer": "This inquiry appears to be a direct transaction query rather than a financial principle question.",
                "retrieved_context": [],
                "retrieval_executed": False,
                "reason": reason
            }
            
        retrieved_docs = self.search_knowledge(query, top_k=3)
        context_str = "\n\n---\n\n".join([f"[Source: {d['source']}]\n{d['content']}" for d in retrieved_docs])
        
        try:
            answer = self.chain.invoke({
                "context": context_str,
                "question": query
            })
        except Exception as e:
            answer = f"Error generating grounded answer: {e}\n\nContext:\n{context_str}"
            
        return {
            "answer": answer,
            "retrieved_context": retrieved_docs,
            "retrieval_executed": True,
            "reason": reason
        }

