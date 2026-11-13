"""
FinSight AI - LLM Factory & LangChain Integration
Initializes Google Gemini via LangChain (ChatGoogleGenerativeAI) with graceful
fallbacks and LangChain LCEL compatibility.
"""
import os
import json
import logging
from typing import Any, Dict, List, Optional, Union
from config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger("FinSightAI.LLMFactory")

# Check LangChain Core availability
try:
    from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
    from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
    from langchain_core.runnables import Runnable, RunnableLambda
    LANGCHAIN_CORE_AVAILABLE = True
except ImportError:
    LANGCHAIN_CORE_AVAILABLE = False

    # Standalone Lightweight Shim for zero-dependency execution and testing
    class BaseMessage:
        def __init__(self, content: str):
            self.content = content
        def __str__(self):
            return self.content

    class HumanMessage(BaseMessage):
        pass

    class AIMessage(BaseMessage):
        pass

    class SystemMessage(BaseMessage):
        pass

    class PromptValue:
        def __init__(self, text: str):
            self.text = text
        def to_string(self) -> str:
            return self.text
        def __str__(self):
            return self.text

    class PromptTemplate:
        def __init__(self, template: str, input_variables: Optional[List[str]] = None):
            self.template = template
            self.input_variables = input_variables or []
            
        @classmethod
        def from_template(cls, template: str):
            import re
            vars_found = re.findall(r'\{([a-zA-Z0-9_]+)\}', template)
            return cls(template=template, input_variables=list(set(vars_found)))
            
        def format(self, **kwargs) -> str:
            return self.template.format(**kwargs)
            
        def invoke(self, input_dict: Dict[str, Any]) -> PromptValue:
            return PromptValue(self.format(**input_dict))
            
        def __or__(self, other):
            return RunnableSequence(self, other)

    class FewShotPromptTemplate(PromptTemplate):
        def __init__(self, examples: List[Dict[str, Any]], example_prompt: Any, prefix: str, suffix: str, input_variables: List[str]):
            self.examples = examples
            self.example_prompt = example_prompt
            self.prefix = prefix
            self.suffix = suffix
            self.input_variables = input_variables
            
        def format(self, **kwargs) -> str:
            lines = [self.prefix]
            for ex in self.examples:
                lines.append(self.example_prompt.format(**ex))
            lines.append(self.suffix.format(**kwargs))
            return "\n\n".join(lines)
            
        def invoke(self, input_dict: Dict[str, Any]) -> PromptValue:
            return PromptValue(self.format(**input_dict))

    class ChatPromptTemplate:
        def __init__(self, messages: List[Any]):
            self.messages = messages
            
        @classmethod
        def from_messages(cls, messages: List[Any]):
            return cls(messages=messages)
            
        def format(self, **kwargs) -> str:
            out = []
            for role, tmpl in self.messages:
                out.append(f"{role.upper()}: {tmpl.format(**kwargs)}")
            return "\n\n".join(out)
            
        def invoke(self, input_dict: Dict[str, Any]) -> PromptValue:
            return PromptValue(self.format(**input_dict))
            
        def __or__(self, other):
            return RunnableSequence(self, other)

    class StrOutputParser:
        def invoke(self, input_obj: Any) -> str:
            if hasattr(input_obj, 'content'):
                return str(input_obj.content)
            return str(input_obj)
            
        def __ror__(self, other):
            return RunnableSequence(other, self)

    class JsonOutputParser:
        def invoke(self, input_obj: Any) -> Dict[str, Any]:
            text = str(input_obj.content) if hasattr(input_obj, 'content') else str(input_obj)
            # Find json block
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*\}|\[.*\])\s*```', text, re.DOTALL)
            if json_match:
                raw_json = json_match.group(1)
            else:
                json_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
                raw_json = json_match.group(1) if json_match else text
            try:
                return json.loads(raw_json)
            except Exception:
                return {"result": text}
                
        def __ror__(self, other):
            return RunnableSequence(other, self)

    class RunnableSequence:
        def __init__(self, first, second):
            self.steps = []
            if isinstance(first, RunnableSequence):
                self.steps.extend(first.steps)
            else:
                self.steps.append(first)
            if isinstance(second, RunnableSequence):
                self.steps.extend(second.steps)
            else:
                self.steps.append(second)
                
        def invoke(self, input_data: Any) -> Any:
            current = input_data
            for step in self.steps:
                if hasattr(step, 'invoke'):
                    current = step.invoke(current)
                elif callable(step):
                    current = step(current)
            return current
            
        def __or__(self, other):
            return RunnableSequence(self, other)


# Fallback Financial Engine when no API key is provided
class FallbackFinancialLLM:
    """
    Intelligent heuristic fallback model that generates realistic financial reasoning
    when an active Gemini API key is missing or quota is exhausted.
    """
    def __init__(self, model_name: str = "offline-gemini-fallback"):
        self.model_name = model_name

    def invoke(self, input_val: Any) -> AIMessage:
        prompt_text = str(input_val.text) if hasattr(input_val, 'text') else str(input_val)
        prompt_lower = prompt_text.lower()
        
        # 1. Intent Detection response
        if "classify the intent" in prompt_lower or "intent classification" in prompt_lower:
            if any(k in prompt_lower for k in ["how much", "spent", "total", "cost", "groceries", "dining", "rent"]):
                intent = "spending_analysis"
            elif any(k in prompt_lower for k in ["find", "search", "lookup", "when did", "statement", "uber", "netflix"]):
                intent = "transaction_lookup"
            elif any(k in prompt_lower for k in ["50/30/20", "budget", "save", "savings", "allocate", "emergency"]):
                intent = "budgeting_and_savings"
            elif any(k in prompt_lower for k in ["advice", "invest", "debt", "avalanche", "snowball", "plan"]):
                intent = "financial_advice"
            else:
                intent = "general_query"
            return AIMessage(content=json.dumps({"intent": intent, "confidence": 0.95, "reasoning": "Heuristic semantic match on financial keywords."}))
            
        # 2. Categorization response
        if "categorize the following bank transaction" in prompt_lower or "assign one category" in prompt_lower:
            cat = "Miscellaneous"
            if any(k in prompt_lower for k in ["payroll", "deposit", "salary", "bonus"]):
                cat = "Income / Payroll"
            elif any(k in prompt_lower for k in ["rent", "mortgage", "apex"]):
                cat = "Housing"
            elif any(k in prompt_lower for k in ["whole foods", "trader joe", "safeway", "costco", "supermarket", "grocer"]):
                cat = "Groceries"
            elif any(k in prompt_lower for k in ["restaurant", "chipotle", "starbucks", "diner", "bistro", "steak"]):
                cat = "Dining Out"
            elif any(k in prompt_lower for k in ["electric", "gas", "water", "utility", "internet", "verizon", "con edison"]):
                cat = "Utilities"
            elif any(k in prompt_lower for k in ["gas station", "shell", "chevron", "uber", "lyft", "transit", "subway"]):
                cat = "Transportation"
            elif any(k in prompt_lower for k in ["pharmacy", "cvs", "walgreens", "health", "doctor", "hospital", "clinic"]):
                cat = "Healthcare"
            elif any(k in prompt_lower for k in ["netflix", "spotify", "gym", "fitness", "cinema", "movie", "concert"]):
                cat = "Entertainment"
            elif any(k in prompt_lower for k in ["amazon", "target", "walmart", "best buy", "clothing"]):
                cat = "Shopping"
            elif any(k in prompt_lower for k in ["vanguard", "fidelity", "index fund", "etf", "invest", "robinhood"]):
                cat = "Investment & Savings"
            return AIMessage(content=json.dumps({"category": cat, "confidence": 0.92}))
            
        # 3. Default contextual advice
        return AIMessage(content=(
            "### FinSight AI Autonomous Analysis\n\n"
            "Based on your current statement data and financial guidelines:\n"
            "- **Cash Flow:** Your income exceeds expenses, indicating positive net cash flow.\n"
            "- **50/30/20 Alignment:** Essential needs are well within sustainable boundaries.\n"
            "- **Recommended Action:** Continue routing unspent cash into high-yield savings to maintain a 3-6 month emergency fund, then maximize tax-advantaged accounts.\n\n"
            "*(Note: Powered by FinSight AI fallback engine. Add your GEMINI_API_KEY for live generative LLM insights)*"
        ))

    def __or__(self, other):
        return RunnableSequence(self, other) if not LANGCHAIN_CORE_AVAILABLE else RunnableLambda(self.invoke) | other


def get_llm(temperature: float = 0.2, api_key: Optional[str] = None):
    """
    Factory function returning a LangChain-compatible Gemini LLM instance.
    Gracefully falls back to FallbackFinancialLLM if key is absent or package is missing.
    """
    active_key = api_key or GEMINI_API_KEY
    
    if not active_key:
        logger.info("No GEMINI_API_KEY configured. Utilizing FinSight AI Fallback LLM.")
        return FallbackFinancialLLM()
        
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=active_key,
            temperature=temperature
        )
        return llm
    except Exception as e:
        logger.warning(f"Could not initialize ChatGoogleGenerativeAI ({e}). Trying google.generativeai...")
        try:
            import google.generativeai as genai
            genai.configure(api_key=active_key)
            # Wrap in simple runnable adapter
            class DirectGenAIWrapper:
                def __init__(self, model_name: str, temp: float):
                    self.model = genai.GenerativeModel(model_name)
                    self.temp = temp
                    
                def invoke(self, input_val: Any) -> AIMessage:
                    prompt_str = str(input_val.text) if hasattr(input_val, 'text') else str(input_val)
                    res = self.model.generate_content(prompt_str)
                    return AIMessage(content=res.text if res.text else "")
                    
                def __or__(self, other):
                    if not LANGCHAIN_CORE_AVAILABLE:
                        return RunnableSequence(self, other)
                    return RunnableLambda(self.invoke) | other
                    
            return DirectGenAIWrapper(GEMINI_MODEL, temperature)
        except Exception as ex:
            logger.error(f"Fallback to internal rule engine due to error: {ex}")
            return FallbackFinancialLLM()

