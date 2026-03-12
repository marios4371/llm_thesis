"""
Enhanced Reasoning Quality Evaluation System for MAS Math Solver
VERSION 9.0: Critic-Based Hypothesis Testing

CHANGELOG v7.3 (over v7.2):
- [NEW] Heterogeneous Model Configuration: each agent role can use a different LLM
- [NEW] AgentRole enum (BASELINE, MATHEMATICIAN, PROGRAMMER, HYPOTHESIS_GENERATOR, JUDGE)
- [NEW] ModelConfig dataclass + HETEROGENEOUS_PRESETS (5 presets)
- [NEW] UnifiedLLMClient accepts model_override parameter
- [NEW] Solver uses _get_client(role) — dispatches to role-specific client
- [NEW] Pipeline supports heterogeneous_preset and custom_config params
- [NEW] CSV output includes model_config per role for experiment tracking
- [FIX] Client deduplication: same (provider, model) pair shares one connection

Inherits v7.1/v7.2 fixes:
- Error response detection, token budget, 429 handling, cache, auth detection
"""

from __future__ import annotations

import os 
import re
import json
import time
import random
import hashlib
import threading
import pickle
import logging
import ast
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from dotenv import load_dotenv

# --- Statistical Libraries Check ---
try:
    from statsmodels.stats.contingency_tables import mcnemar
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from scipy.stats import chi2
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: 'datasets' library not found. Standard curated set will be used.")

# --- LLM Providers ---
from openai import OpenAI

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

# --- [NEW v8.0] Symbolic Solver ---
try:
    import sympy
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    print("Warning: 'sympy' not found. Symbolic solver fallback disabled. pip install sympy")


# --------------------------- Configuration ---------------------------

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# ========================== [NEW v7.3] Heterogeneous Model Config ==========================

class AgentRole(Enum):
    """
    Each agent role in the MAS-SHT pipeline can be assigned
    to a different provider + model combination.
    """
    BASELINE = "baseline"
    MATHEMATICIAN = "mathematician"
    PROGRAMMER = "programmer"
    HYPOTHESIS_GENERATOR = "hypothesis_generator"
    JUDGE = "judge"

@dataclass
class ModelConfig:
    """
    Configuration for a single model endpoint.
    
    provider: "groq" or "google"
    model_name: specific model string (e.g., "llama-3.3-70b-versatile", "gemma2-9b-it")
                If None, uses the provider's default.
    """
    provider: str = "groq"
    model_name: Optional[str] = None  # None = use provider default


# Pre-defined heterogeneous configurations
# Users can also build custom configs

HETEROGENEOUS_PRESETS: Dict[str, Dict[AgentRole, ModelConfig]] = {
    # All roles use the same model (backward compatible, same as v7.2)
    "homogeneous_groq": {
        AgentRole.BASELINE:              ModelConfig("groq", "llama-3.3-70b-versatile"),
        AgentRole.MATHEMATICIAN:         ModelConfig("groq", "llama-3.3-70b-versatile"),
        AgentRole.PROGRAMMER:            ModelConfig("groq", "llama-3.3-70b-versatile"),
        AgentRole.HYPOTHESIS_GENERATOR:  ModelConfig("groq", "llama-3.3-70b-versatile"),
        AgentRole.JUDGE:                 ModelConfig("groq", "llama-3.3-70b-versatile"),
    },
    
    # Cross-architecture diversity: different model families per role
    # Key insight: diversity in model architecture → diversity in reasoning patterns
    "diverse_groq": {
        AgentRole.BASELINE:              ModelConfig("groq", "gemma2-9b-it"),              # Different family for baseline diversity
        AgentRole.MATHEMATICIAN:         ModelConfig("groq", "llama-3.3-70b-versatile"),   # Best reasoning for blueprint
        AgentRole.PROGRAMMER:            ModelConfig("groq", "llama-3.3-70b-versatile"),   # Needs precise instruction following
        AgentRole.HYPOTHESIS_GENERATOR:  ModelConfig("groq", "mixtral-8x7b-32768"),        # MoE architecture → diverse strategies
        AgentRole.JUDGE:                 ModelConfig("groq", "llama-3.3-70b-versatile"),   # Needs strong evaluation
    },
    
    # Cross-provider diversity: use both Groq and Google
    "cross_provider": {
        AgentRole.BASELINE:              ModelConfig("google", None),                       # Gemini as independent baseline
        AgentRole.MATHEMATICIAN:         ModelConfig("groq", "llama-3.3-70b-versatile"),   # LLaMA for structured JSON output
        AgentRole.PROGRAMMER:            ModelConfig("groq", "llama-3.3-70b-versatile"),   # LLaMA for code generation
        AgentRole.HYPOTHESIS_GENERATOR:  ModelConfig("google", None),                       # Gemini for diverse strategies
        AgentRole.JUDGE:                 ModelConfig("groq", "llama-3.3-70b-versatile"),   # LLaMA for final judgment
    },
    
    # Budget-optimized: small models where possible, large only where critical
    "budget_optimized": {
        AgentRole.BASELINE:              ModelConfig("groq", "llama-3.1-8b-instant"),      # Fast & cheap baseline
        AgentRole.MATHEMATICIAN:         ModelConfig("groq", "llama-3.3-70b-versatile"),   # Full power for blueprint
        AgentRole.PROGRAMMER:            ModelConfig("groq", "llama-3.1-8b-instant"),      # Small model can follow blueprints
        AgentRole.HYPOTHESIS_GENERATOR:  ModelConfig("groq", "llama-3.3-70b-versatile"),   # Needs creativity
        AgentRole.JUDGE:                 ModelConfig("groq", "llama-3.3-70b-versatile"),   # Needs strong judgment
    },
    
    # Homogeneous Google
    "homogeneous_google": {
        AgentRole.BASELINE:             ModelConfig("google", "gemini-2.5-flash-lite"),
        AgentRole.MATHEMATICIAN:         ModelConfig("google", "gemini-2.5-flash-lite"),
        AgentRole.PROGRAMMER:            ModelConfig("google", "gemini-2.5-flash-lite"),
        AgentRole.HYPOTHESIS_GENERATOR:  ModelConfig("google", "gemini-2.5-flash-lite"),
        AgentRole.JUDGE:                 ModelConfig("google", "gemini-2.5-flash-lite"),
    },
}


# --------------------------- Cache & Logging ---------------------------

CACHE_FILE = "call_cache_v6.pkl"
CALL_CACHE: Dict[str, Any] = {}

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("MAS_Pipeline")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(ch)
    return logger

logger = setup_logging()

# --------------------------- Helpers ---------------------------

def _make_cache_key(provider: str, model_name: str, messages: List[Dict[str, str]], temperature: float) -> str:
    payload = {"provider": provider, "model": model_name, "messages": messages, "temperature": temperature}
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

class RateLimiter:
    """Enhanced rate limiter with 429-specific backoff and token tracking."""
    def __init__(self, requests_per_minute: int = 12):
        self.delay = 60.0 / max(1, requests_per_minute)
        self.last_call = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
            self.last_call = time.time()


class TokenBudget:
    """
    [NEW v7.2] Tracks estimated token usage against daily limit.
    Groq free tier: 100,000 tokens/day (TPD).
    """
    def __init__(self, daily_limit: int = 100_000):
        self.daily_limit = daily_limit
        self.tokens_used = 0
        self.lock = threading.Lock()
        self._warning_issued = False
    
    def estimate_tokens(self, messages: List[Dict[str, str]], max_tokens: int) -> int:
        """Realistic estimate: input_chars/4 + max_tokens*0.35 (models rarely use full max)."""
        input_chars = sum(len(m.get("content", "")) for m in messages)
        estimated_input = input_chars // 4
        estimated_output = int(max_tokens * 0.35)
        return estimated_input + estimated_output
    
    def record_usage(self, estimated_tokens: int) -> None:
        with self.lock:
            self.tokens_used += estimated_tokens
    
    def can_afford(self, estimated_tokens: int) -> bool:
        with self.lock:
            remaining = self.daily_limit - self.tokens_used
            if remaining < estimated_tokens:
                if not self._warning_issued:
                    logger.warning(
                        f"TOKEN BUDGET: ~{self.tokens_used:,} used of {self.daily_limit:,} daily limit. "
                        f"Need ~{estimated_tokens:,} but only ~{remaining:,} remaining."
                    )
                    self._warning_issued = True
                return False
            return True
    
    def remaining(self) -> int:
        with self.lock:
            return max(0, self.daily_limit - self.tokens_used)
    
    def usage_report(self) -> str:
        pct = (self.tokens_used / self.daily_limit) * 100
        return (f"Token usage: ~{self.tokens_used:,} / {self.daily_limit:,} "
                f"({pct:.1f}%) | ~{self.remaining():,} remaining")

groq_limiter = RateLimiter(requests_per_minute=30)   # Groq free tier allows 30 RPM
google_limiter = RateLimiter(requests_per_minute=15)
token_budget = TokenBudget(daily_limit=100_000)  # [NEW v7.2] Groq free tier

def _safe_json_load(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None


# ==========================================================================
# [FIX v7.1] Error Detection Helper
# ==========================================================================

def _is_error_response(text: Any) -> bool:
    """
    Detect whether an LLM response is actually an error message.
    This prevents HTTP status codes (401, 429, 500, etc.) from being
    parsed as numeric answers.
    """
    if text is None:
        return True
    s = str(text).strip()
    if s.startswith("ERROR_GENERATION"):
        return True
    if s.startswith("ERROR_"):  # Catches ERROR_AUTH_401, ERROR_RATE_LIMIT_DAILY, ERROR_BUDGET_EXCEEDED
        return True
    # Check for common API error patterns
    error_patterns = [
        r"error.*(?:401|403|429|500|502|503)",
        r"(?:unauthorized|forbidden|rate.?limit|internal.?server)",
        r"authentication.*(?:failed|error|invalid)",
        r"api.?key.*(?:invalid|missing|expired)",
        r"token.?limit.*(?:reached|exceeded)",
        r"budget.*exceeded",
    ]
    s_lower = s.lower()
    for pattern in error_patterns:
        if re.search(pattern, s_lower):
            return True
    # Too short to be a real response (likely error)
    if len(s) < 5 and not re.match(r'^-?\d+\.?\d*$', s):
        return True
    return False


# OPTIMIZED: Better blueprint extraction with structured fallback
def _extract_blueprint_json(text: str) -> dict:
    """
    Enhanced JSON extraction with better fallback handling.
    Ensures required keys exist even if parsing fails.
    """
    # [FIX v7.1] Check for error response first
    if _is_error_response(text):
        logger.warning(f"Mathematician returned error response: {str(text)[:200]}")
        return {
            "unknown": "the answer",
            "givens": {},
            "solution_steps": ["Error: LLM call failed"],
            "equations": [],
            "distractor_check": "",
            "metamorphic_tests": [],
            "notes": f"ERROR: {str(text)[:200]}"
        }
    
    text = str(text).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    
    # Try direct parse
    try:
        bp = json.loads(text)
        if isinstance(bp, dict):
            bp.setdefault("unknown", "the answer")
            bp.setdefault("givens", {})
            bp.setdefault("solution_steps", [])
            bp.setdefault("equations", [])
            bp.setdefault("distractor_check", "")
            bp.setdefault("metamorphic_tests", [])
            bp.setdefault("notes", "")
            return bp
    except:
        pass
    
    # Try substring extraction
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            bp = json.loads(text[start:end+1])
            if isinstance(bp, dict):
                bp.setdefault("unknown", "the answer")
                bp.setdefault("givens", {})
                bp.setdefault("solution_steps", [])
                bp.setdefault("equations", [])
                bp.setdefault("distractor_check", "")
                bp.setdefault("metamorphic_tests", [])
                bp.setdefault("notes", "")
                return bp
        except:
            pass
    
    # Fallback: extract what we can
    givens = {}
    equations = []
    
    # Try to extract givens dict
    givens_match = re.search(r'"givens"\s*:\s*(\{[^}]+\})', text, re.DOTALL)
    if givens_match:
        try:
            givens = json.loads(givens_match.group(1))
        except:
            pass
    
    # Try to extract equations array
    eqs_match = re.search(r'"equations"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if eqs_match:
        try:
            equations = json.loads(f'[{eqs_match.group(1)}]')
        except:
            pass
    
    return {
        "unknown": "the answer",
        "givens": givens,
        "solution_steps": ["Solve step by step"],
        "equations": equations,
        "distractor_check": "",
        "metamorphic_tests": [],
        "notes": text[:800]
    }

def _extract_givens_dict_from_code(code_str: str) -> Optional[dict]:
    m = re.search(r"givens\s*=\s*(\{.*?\})\s*(?:\n|$)", code_str, re.DOTALL)
    if not m:
        return None
    try:
        return ast.literal_eval(m.group(1))
    except Exception:
        return None

def _replace_givens_dict_in_code(code_str: str, new_givens: dict) -> str:
    m = re.search(r"(givens\s*=\s*)(\{.*?\})(\s*(?:\n|$))", code_str, re.DOTALL)
    if not m:
        return code_str
    prefix, _, suffix = m.group(1), m.group(2), m.group(3)
    return code_str[:m.start()] + prefix + repr(new_givens) + suffix + code_str[m.end():]

# --------------------------- Robust Code Extractor ---------------------------

def _extract_code_from_response(raw: str) -> Optional[str]:
    """
    Enhanced code extraction with multiple pattern matching strategies.
    """
    # [FIX v7.1] Don't try to extract code from error responses
    if _is_error_response(raw):
        return None
    
    s = str(raw)
    
    # Strategy 1: Standard markdown fences
    patterns = [
        r"```python\s+(.*?)```",
        r"```py\s+(.*?)```",
        r"```\s+(.*?)```",
        r"~~~python\s+(.*?)~~~",
        r"~~~\s+(.*?)~~~",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, s, re.DOTALL | re.IGNORECASE)
        if match:
            code = match.group(1).strip()
            code = re.sub(r"^python\s*\n", "", code, flags=re.IGNORECASE)
            return code
    
    # Strategy 2: Open fence (missing closing)
    open_patterns = [
        r"```(?:python|py)?\s+(.*?)$",
        r"~~~(?:python|py)?\s+(.*?)$",
    ]
    for pattern in open_patterns:
        match = re.search(pattern, s, re.DOTALL | re.IGNORECASE)
        if match:
            code = match.group(1).strip()
            code = re.split(r"\n\n(?:ANSWER|---|Note|Explanation)", code, maxsplit=1)[0]
            return code.strip()
    
    # Strategy 3: Code starts with "givens = " (no fence at all)
    givens_match = re.search(r"^(givens\s*=\s*\{.*)", s, re.DOTALL | re.MULTILINE)
    if givens_match:
        code = givens_match.group(1)
        code = re.split(r"\n\n(?:ANSWER|---)", code, maxsplit=1)[0]
        return code.strip()
    
    return None


def _extract_last_number(text: str) -> Optional[float]:
    """
    Extract the last numeric value from text, handling various formats.
    """
    # [FIX v7.1] Don't extract numbers from error responses
    if _is_error_response(text):
        return None
    
    text = str(text).strip()
    
    # Remove common non-numeric suffixes
    text = re.sub(r'\s*(dollars?|cents?|units?|items?|people|apples?|hours?|minutes?|days?|years?)\s*$', 
                  '', text, flags=re.IGNORECASE)
    
    # Find all numbers (including negatives, decimals, with commas)
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    
    if not numbers:
        return None
    
    last_num = numbers[-1].replace(',', '')
    try:
        return float(last_num)
    except:
        return None


def _format_blueprint_for_programmer(bp: dict) -> str:
    """
    Format blueprint in a clear, structured way for the Programmer to follow.
    """
    unknown = bp.get("unknown", "the answer")
    givens = bp.get("givens", {})
    solution_steps = bp.get("solution_steps", [])
    equations = bp.get("equations", [])
    distractor_check = bp.get("distractor_check", "")
    
    blueprint_text = f"""TARGET: {unknown}

GIVEN VALUES:
{json.dumps(givens, indent=2)}

SOLUTION STEPS:
{chr(10).join([f"{i+1}. {step}" for i, step in enumerate(solution_steps)])}

EQUATIONS TO IMPLEMENT (in order):
{chr(10).join([f"  {eq}" for eq in equations])}"""

    if distractor_check and distractor_check != "None":
        blueprint_text += f"\n\nDISTRACTORS TO IGNORE:\n{distractor_check}"
    
    return blueprint_text


# --------------------------- Code Executor ---------------------------

class PythonExecutor:
    @staticmethod
    def execute(code_str: str) -> Tuple[bool, str]:
        """Execute Python code safely with better error messages."""
        forbidden = [
            "import os", "import sys", "subprocess", "__import__",
            "eval(", "exec(", "compile(", "open(", "file(",
            "input(", "raw_input(", "rm -rf", "rmdir"
        ]
        
        code_lower = code_str.lower()
        for token in forbidden:
            if token in code_lower:
                return False, f"SecurityError: Forbidden token '{token}'"
        
        try:
            local_vars = {}
            import io
            from contextlib import redirect_stdout
            
            buf = io.StringIO()
            with redirect_stdout(buf):
                exec(code_str, {"__builtins__": __builtins__}, local_vars)
            
            output = buf.getvalue().strip()
            
            if not output:
                if 'answer' in local_vars:
                    return True, str(local_vars['answer'])
                elif 'result' in local_vars:
                    return True, str(local_vars['result'])
                else:
                    return False, "NoOutput: Code produced no output or answer variable"
            
            return True, output
            
        except NameError as e:
            return False, f"NameError: {str(e)}. Check variable definitions."
        except KeyError as e:
            return False, f"KeyError: {str(e)}. Check givens dict keys."
        except ZeroDivisionError:
            return False, "ZeroDivisionError: Cannot divide by zero"
        except Exception as e:
            return False, f"ExecutionError: {str(e)}"


# ==========================================================================
# [NEW v8.0] Symbolic Solver Fallback (SymPy)
# ==========================================================================

class SymbolicSolver:
    """
    When the Programmer's code execution fails, attempt to solve the
    blueprint equations symbolically using SymPy.
    
    This eliminates arithmetic errors entirely by delegating computation
    to a computer algebra system. Works best for problems expressible as
    algebraic equations (linear, polynomial, rate/proportion).
    
    Flow:
        1. Extract givens dict and equations from blueprint
        2. Substitute givens into equations
        3. Detect if there's an unknown variable to solve for
        4. Execute the equation chain symbolically
        5. Return numeric answer
    """

    @staticmethod
    def solve_from_blueprint(blueprint: dict) -> Tuple[bool, str, str]:
        """
        Attempt to solve a blueprint's equations using SymPy.
        
        Returns:
            (success: bool, answer: str, trace: str)
        """
        if not SYMPY_AVAILABLE:
            return False, "unknown", "SymPy not installed"
        
        givens = blueprint.get("givens", {})
        equations = blueprint.get("equations", [])
        
        if not equations:
            return False, "unknown", "No equations in blueprint"
        
        trace_lines = ["[SymPy Symbolic Solver]"]
        
        try:
            # Build a namespace with givens values
            namespace = {}
            givens_dict = {}
            
            for key, val in givens.items():
                if isinstance(val, (int, float)):
                    namespace[key] = val
                    givens_dict[key] = val
                    trace_lines.append(f"  Given: {key} = {val}")
            
            # Make givens accessible as dict too (for givens['key'] syntax)
            namespace['givens'] = givens_dict
            
            # Execute each equation in order using Python eval with restricted builtins
            safe_builtins = {
                "abs": abs, "round": round, "min": min, "max": max,
                "int": int, "float": float, "sum": sum, "len": len,
                "pow": pow, "divmod": divmod,
            }
            
            # Add math functions
            import math
            for fn_name in ['ceil', 'floor', 'sqrt', 'log', 'log10', 'exp', 'pi']:
                if hasattr(math, fn_name):
                    safe_builtins[fn_name] = getattr(math, fn_name)
            
            exec_globals = {"__builtins__": safe_builtins, "givens": givens_dict}
            exec_locals = dict(namespace)
            
            last_result = None
            for eq in equations:
                eq = eq.strip()
                if not eq or eq.startswith("#"):
                    continue
                
                trace_lines.append(f"  Exec: {eq}")
                
                try:
                    exec(eq, exec_globals, exec_locals)
                    # Track the last assigned variable
                    if "=" in eq and not eq.strip().startswith("if"):
                        var_name = eq.split("=")[0].strip()
                        if var_name in exec_locals:
                            last_result = exec_locals[var_name]
                except Exception as eq_err:
                    trace_lines.append(f"  ERROR in equation: {eq_err}")
                    # Try SymPy symbolic evaluation as last resort
                    sympy_result = SymbolicSolver._try_sympy_eval(eq, exec_locals, givens_dict)
                    if sympy_result is not None:
                        var_name = eq.split("=")[0].strip()
                        exec_locals[var_name] = sympy_result
                        last_result = sympy_result
                        trace_lines.append(f"  SymPy resolved: {var_name} = {sympy_result}")
                    else:
                        return False, "unknown", "\n".join(trace_lines)
            
            # Get final answer
            answer = exec_locals.get("answer", last_result)
            if answer is None:
                return False, "unknown", "\n".join(trace_lines) + "\n  No 'answer' variable found"
            
            # Convert to float
            try:
                answer_float = float(answer)
                trace_lines.append(f"  RESULT: {answer_float}")
                return True, str(answer_float), "\n".join(trace_lines)
            except (ValueError, TypeError):
                return False, "unknown", "\n".join(trace_lines) + f"\n  Non-numeric answer: {answer}"
            
        except Exception as e:
            trace_lines.append(f"  FATAL: {type(e).__name__}: {e}")
            return False, "unknown", "\n".join(trace_lines)

    @staticmethod
    def _try_sympy_eval(equation_str: str, local_vars: dict, givens: dict) -> Optional[float]:
        """
        Try to evaluate a single equation using SymPy when Python exec fails.
        Handles cases like division expressions, fractional arithmetic, etc.
        """
        if not SYMPY_AVAILABLE:
            return None
        
        try:
            # Extract RHS of assignment
            if "=" not in equation_str:
                return None
            
            parts = equation_str.split("=", 1)
            rhs = parts[1].strip()
            
            # Replace givens['key'] with actual values
            for key, val in givens.items():
                rhs = rhs.replace(f"givens['{key}']", str(val))
                rhs = rhs.replace(f'givens["{key}"]', str(val))
            
            # Replace known local variables
            for key, val in local_vars.items():
                if isinstance(val, (int, float)) and key != "givens":
                    # Only replace whole words
                    rhs = re.sub(rf'\b{re.escape(key)}\b', str(val), rhs)
            
            # Parse and evaluate with SymPy
            transformations = standard_transformations + (implicit_multiplication_application,)
            expr = parse_expr(rhs, transformations=transformations)
            result = float(expr.evalf())
            
            return result
        except Exception:
            return None


# ==========================================================================
# [FIX v7.1] Custom Exception for API Failures
# ==========================================================================

class LLMCallError(Exception):
    """Raised when all retries for an LLM API call are exhausted."""
    pass


# --------------------------- Unified LLM Client ---------------------------

class UnifiedLLMClient:
    def __init__(self, provider: str = "groq", use_cache: bool = False, model_override: Optional[str] = None):
        """
        [UPDATED v7.3] Accepts optional model_override to specify exact model.
        This enables heterogeneous configurations where different agent roles
        use different models from the same or different providers.
        """
        self.provider = provider
        self.use_cache = use_cache
        self.model_name = "unknown"
        self.limiter = groq_limiter if provider == "groq" else google_limiter

        if use_cache and os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "rb") as f:
                global CALL_CACHE
                try:
                    CALL_CACHE = pickle.load(f)
                except:
                    CALL_CACHE = {}

        if provider == "groq":
            if not GROQ_API_KEY: raise ValueError("Missing GROQ_API_KEY in .env file")
            self.client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
            # [v7.3] Use model_override if provided, else default
            self.model_name = model_override or "llama-3.3-70b-versatile"
        elif provider == "google":
            if not GOOGLE_API_KEY: raise ValueError("Missing GOOGLE_API_KEY in .env file")
            if not GOOGLE_AVAILABLE: raise ImportError("Google SDK missing. pip install google-generativeai")
            genai.configure(api_key=GOOGLE_API_KEY)
            if model_override:
                self.model_name = model_override
                self.client = genai.GenerativeModel(model_override)
            else:
                self._setup_google_model()
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def __repr__(self) -> str:
        return f"LLMClient({self.provider}/{self.model_name})"

    # [FIX v7.1] Validate API key at startup
    def validate_connection(self) -> bool:
        """Test that the API key works before running the full pipeline."""
        logger.info(f"Validating {self.provider} API connection...")
        try:
            test_response = self.call_model(
                [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
                temperature=0.0,
                max_tokens=50
            )
            if _is_error_response(test_response):
                logger.error(f"API validation FAILED. Response: {test_response}")
                return False
            logger.info(f"API validation OK. Test response: {str(test_response)[:100]}")
            return True
        except Exception as e:
            logger.error(f"API validation FAILED with exception: {e}")
            return False

    def call_model(self, messages: List[Dict[str, str]], temperature: float = 0.3, max_tokens: int = 1200) -> Any:
        key = _make_cache_key(self.provider, self.model_name, messages, temperature)

        if self.use_cache and key in CALL_CACHE:
            return CALL_CACHE[key]

        # [FIX v7.2] Check token budget before calling
        estimated = token_budget.estimate_tokens(messages, max_tokens)
        if not token_budget.can_afford(estimated):
            return "ERROR_BUDGET_EXCEEDED: Daily token limit reached. Wait 24h or upgrade to Dev tier."

        last_err = None

        def _call_once():
            self.limiter.wait()
            if self.provider == "groq":
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                # [FIX v7.2] Track actual token usage from response
                if hasattr(resp, 'usage') and resp.usage:
                    actual_tokens = getattr(resp.usage, 'total_tokens', estimated)
                    token_budget.record_usage(actual_tokens)
                else:
                    token_budget.record_usage(estimated)
                return resp.choices[0].message.content
            if self.provider == "google":
                sys_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
                user_prompt = "\n\n".join([m["content"] for m in messages if m["role"] != "system"])
                full_prompt = f"System:\n{sys_prompt}\n\nTask:\n{user_prompt}"
                resp = self.client.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens
                    )
                )
                token_budget.record_usage(estimated)
                return getattr(resp, "text", "")

        for attempt in range(6):
            try:
                res = _call_once()
                
                if res is None or str(res).strip() == "":
                    last_err = "Empty response from API"
                    logger.warning(f"Attempt {attempt+1}/6: Empty response. Retrying...")
                    time.sleep(min(12.0, 1.5 * (attempt + 1)))
                    continue
                
                if self.use_cache:
                    CALL_CACHE[key] = res
                    with open(CACHE_FILE, "wb") as f:
                        pickle.dump(CALL_CACHE, f)
                return res
            except Exception as e:
                last_err = str(e)
                err_str = str(e).lower()
                logger.warning(f"Attempt {attempt+1}/6 failed: {type(e).__name__}: {str(e)[:200]}")
                
                # [FIX v7.1] Auth errors — no point retrying
                if "401" in err_str or "unauthorized" in err_str or "authentication" in err_str:
                    logger.error("AUTHENTICATION ERROR: API key is invalid or expired.")
                    return f"ERROR_AUTH_401: {last_err}"
                
                if "403" in err_str or "forbidden" in err_str:
                    logger.error("FORBIDDEN: API key does not have access to this model.")
                    return f"ERROR_AUTH_403: {last_err}"
                
                # [FIX v7.2] 429 Rate Limit — extract wait time from error message
                if "429" in err_str or "rate_limit" in err_str or "rate limit" in err_str:
                    # Try to extract wait time from Groq error (e.g., "try again in 8m27.168s")
                    wait_match = re.search(r'try again in (\d+)m([\d.]+)s', str(e))
                    if wait_match:
                        wait_minutes = int(wait_match.group(1))
                        wait_seconds = float(wait_match.group(2))
                        total_wait = wait_minutes * 60 + wait_seconds + 5  # +5s buffer
                        
                        if total_wait > 600:  # More than 10 minutes = daily limit hit
                            logger.error(
                                f"DAILY TOKEN LIMIT REACHED. Groq says wait {wait_minutes}m{wait_seconds:.0f}s. "
                                f"This usually means you've hit the 100K tokens/day free tier limit. "
                                f"Options: (1) Wait until tomorrow, (2) Upgrade to Dev tier at console.groq.com"
                            )
                            return f"ERROR_RATE_LIMIT_DAILY: {last_err}"
                        
                        logger.info(f"Rate limited. Waiting {total_wait:.0f}s as requested by Groq...")
                        time.sleep(total_wait)
                        continue
                    else:
                        # Generic 429 — exponential backoff with jitter
                        backoff = min(120, (2 ** attempt) * 5 + random.uniform(0, 5))
                        logger.info(f"Rate limited (429). Backing off {backoff:.0f}s...")
                        time.sleep(backoff)
                        continue
                
                # Other errors — standard backoff
                time.sleep(min(12.0, 1.5 * (attempt + 1)))

        return f"ERROR_GENERATION: {last_err or 'unknown_error'}"

    def _setup_google_model(self):
        try:
            available = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
            target = next((m for m in available if "flash" in m), available[0] if available else "models/gemini-1.5-flash")
            self.model_name = target
            self.client = genai.GenerativeModel(target)
        except Exception:
            self.model_name = "gemini-1.5-flash"
            self.client = genai.GenerativeModel(self.model_name)

# --------------------------- Dataset Manager ---------------------------

class EnhancedProblemManager:
    def __init__(self, random_seed: Optional[int] = None):
        random.seed(random_seed)

    def _maybe_harden(self, problem: str, hardener: Optional[str]) -> str:
        if not hardener:
            return problem

        if hardener == "distractor":
            names = ["Alex", "Maria", "Nikos", "Elena", "Chris", "Sofia"]
            items = ["stickers", "marbles", "notebooks", "coins", "candies", "tickets"]
            n1 = random.randint(7, 99)
            n2 = random.randint(10, 250)
            n3 = random.randint(2, 60)
            who = random.choice(names)
            it = random.choice(items)

            distractors = [
                f"Unrelated note: {who} counted {n1} {it} yesterday, but that does not affect the question.",
                f"Extra context (ignore): A different store sold {n2} items in total last week.",
                f"Reminder: The number {n3} appears in a separate example and is irrelevant here.",
            ]
            k = random.choice([1, 2, 3])
            return problem.strip() + "\n\n" + "\n".join(distractors[:k])

        return problem

    def load_random_problems(self, datasets_list: List[str], num_problems: int, hardener: Optional[str] = None) -> List[Dict[str, str]]:
        pool: List[Dict[str, str]] = []

        if not DATASETS_AVAILABLE:
            curated = [
                {"id": "c1", "puzzle": "If 2x + 3 = 15, what is x?", "answer": "6", "dataset": "curated"},
                {"id": "c2", "puzzle": "Jane has 5 apples. She eats 2. How many left?", "answer": "3", "dataset": "curated"},
                {"id": "c3", "puzzle": "Calculate 15% of 200.", "answer": "30", "dataset": "curated"},
                {"id": "c4", "puzzle": "Solve for x: x^2 - 4 = 0 (positive root)", "answer": "2", "dataset": "curated"},
                {"id": "c5", "puzzle": "A train travels 60 mph for 2 hours. Distance?", "answer": "120", "dataset": "curated"},
            ]
            for c in curated[:num_problems]:
                c["puzzle"] = self._maybe_harden(c["puzzle"], hardener)
                pool.append(c)
            return pool[:num_problems]

        if not datasets_list:
            datasets_list = ["gsm8k_test"]
        per_ds = max(1, (num_problems + len(datasets_list) - 1) // len(datasets_list))

        for ds_name in datasets_list:
            ds_name_norm = ds_name.strip().lower()

            try:
                if ds_name_norm in ["gsm8k", "gsm8k_test", "gsm8k-train", "gsm8k_train"]:
                    split = "test" if ds_name_norm in ["gsm8k", "gsm8k_test"] else "train"
                    ds = load_dataset("openai/gsm8k", "main", split=split)
                    idxs = random.sample(range(len(ds)), min(len(ds), per_ds * 3))
                    for i in idxs[:per_ds]:
                        q = ds[i].get("question", "")
                        a = ds[i].get("answer", "")
                        item = {
                            "puzzle": self._maybe_harden(q, hardener),
                            "answer": a,
                            "dataset": f"gsm8k_{split}",
                            "id": f"gsm8k_{split}_{i}",
                        }
                        pool.append(item)

                elif ds_name_norm in ["gsm-hard", "gsm_hard"]:
                    ds = load_dataset("reasoning-machines/gsm-hard", split="train")
                    idxs = random.sample(range(len(ds)), min(len(ds), per_ds * 3))
                    for i in idxs[:per_ds]:
                        q = ds[i].get("input", "")
                        a = ds[i].get("target", "")
                        item = {
                            "puzzle": self._maybe_harden(q, hardener),
                            "answer": a,
                            "dataset": "gsm-hard",
                            "id": f"gsm-hard_{i}",
                        }
                        pool.append(item)

                elif ds_name_norm in ["gsm-plus", "gsm_plus", "gsmplus"]:
                    ds = load_dataset("qintongli/GSM-Plus", split="test")
                    idxs = random.sample(range(len(ds)), min(len(ds), per_ds * 3))
                    for i in idxs[:per_ds]:
                        q = ds[i].get("question", "")
                        a = ds[i].get("answer", "")
                        pt = ds[i].get("perturbation_type", "")
                        item = {
                            "puzzle": self._maybe_harden(q, hardener),
                            "answer": a,
                            "dataset": f"gsm-plus:{pt}" if pt else "gsm-plus",
                            "id": f"gsm-plus_{i}",
                        }
                        pool.append(item)

                elif ds_name_norm in ["gsm-symbolic", "gsm_symbolic", "gsm-symbolic-main", "gsm_symbolic_main"]:
                    ds = load_dataset("apple/GSM-Symbolic", name="main", split="test")
                    idxs = random.sample(range(len(ds)), min(len(ds), per_ds * 3))
                    for i in idxs[:per_ds]:
                        q = ds[i].get("question", "")
                        a = ds[i].get("answer", "")
                        item = {
                            "puzzle": self._maybe_harden(q, hardener),
                            "answer": a,
                            "dataset": "gsm-symbolic:main",
                            "id": f"gsm-symbolic_main_{i}",
                        }
                        pool.append(item)

                elif ds_name_norm in ["gsm-symbolic-p1", "gsm_symbolic_p1"]:
                    ds = load_dataset("apple/GSM-Symbolic", name="p1", split="test")
                    idxs = random.sample(range(len(ds)), min(len(ds), per_ds * 3))
                    for i in idxs[:per_ds]:
                        q = ds[i].get("question", "")
                        a = ds[i].get("answer", "")
                        item = {
                            "puzzle": self._maybe_harden(q, hardener),
                            "answer": a,
                            "dataset": "gsm-symbolic:p1",
                            "id": f"gsm-symbolic_p1_{i}",
                        }
                        pool.append(item)

                elif ds_name_norm in ["gsm-symbolic-p2", "gsm_symbolic_p2"]:
                    ds = load_dataset("apple/GSM-Symbolic", name="p2", split="test")
                    idxs = random.sample(range(len(ds)), min(len(ds), per_ds * 3))
                    for i in idxs[:per_ds]:
                        q = ds[i].get("question", "")
                        a = ds[i].get("answer", "")
                        item = {
                            "puzzle": self._maybe_harden(q, hardener),
                            "answer": a,
                            "dataset": "gsm-symbolic:p2",
                            "id": f"gsm-symbolic_p2_{i}",
                        }
                        pool.append(item)

                elif ds_name_norm in ["svamp"]:
                    ds = load_dataset("ChilleD/SVAMP", split="test")
                    idxs = random.sample(range(len(ds)), min(len(ds), per_ds * 3))
                    for i in idxs[:per_ds]:
                        q = ds[i].get("question_concat", None)
                        if not q:
                            body = ds[i].get("Body", "")
                            ques = ds[i].get("Question", ds[i].get("question", ""))
                            q = (str(body).strip() + "\n" + str(ques).strip()).strip()
                        a = ds[i].get("Answer", ds[i].get("answer", ""))
                        item = {
                            "puzzle": self._maybe_harden(str(q), hardener),
                            "answer": a,
                            "dataset": "svamp:test",
                            "id": f"svamp_test_{i}",
                        }
                        pool.append(item)

                else:
                    logger.warning(f"Unknown dataset key: {ds_name}. Skipping.")

            except Exception as e:
                logger.warning(f"Failed to load dataset '{ds_name}': {e}")

        if len(pool) < num_problems:
            curated = [
                {"id": "c1", "puzzle": "If 2x + 3 = 15, what is x?", "answer": "6", "dataset": "curated"},
                {"id": "c2", "puzzle": "Jane has 5 apples. She eats 2. How many left?", "answer": "3", "dataset": "curated"},
                {"id": "c3", "puzzle": "Calculate 15% of 200.", "answer": "30", "dataset": "curated"},
                {"id": "c4", "puzzle": "Solve for x: x^2 - 4 = 0 (positive root)", "answer": "2", "dataset": "curated"},
                {"id": "c5", "puzzle": "A train travels 60 mph for 2 hours. Distance?", "answer": "120", "dataset": "curated"},
            ]
            for c in curated:
                if len(pool) >= num_problems:
                    break
                c2 = dict(c)
                c2["puzzle"] = self._maybe_harden(c2["puzzle"], hardener)
                pool.append(c2)

        random.shuffle(pool)
        return pool[:num_problems]


# --------------------------- Solver (Architect-Engineer Pattern) ---------------------------

@dataclass
class AgentResponse:
    agent: str
    answer: str
    parsed: Any
    confidence: float
    reasoning_trace: str
    quality_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HypothesisResult:
    hypothesis_id: str
    strategy_name: str
    blueprint: dict
    code: Optional[str]
    code_success: bool
    execution_output: str
    answer: str
    parsed_answer: Optional[float]
    confidence: float
    agent_response: Optional[AgentResponse]

@dataclass
class HypothesisLog:
    problem: str
    expected: str
    candidates: List[HypothesisResult] = field(default_factory=list)
    triage_result: Optional[str] = None
    judge_reasoning: Optional[str] = None
    final_answer: str = "unknown"
    final_strategy: str = "none"
    hypothesis_testing_triggered: bool = False
    api_calls_used: int = 3

class QualityEnhancedMultiAgentSolver:
    
    def __init__(self, client: UnifiedLLMClient = None,
                 clients: Dict[AgentRole, UnifiedLLMClient] = None):
        """
        [UPDATED v7.3] Supports heterogeneous model configuration.
        
        Args:
            client: Single client for all roles (backward compatible).
            clients: Dict mapping AgentRole → UnifiedLLMClient.
                     If both provided, 'clients' takes precedence.
                     Missing roles in 'clients' fall back to 'client'.
        """
        # Build role→client mapping
        self._clients: Dict[AgentRole, UnifiedLLMClient] = {}
        
        if clients:
            self._clients = dict(clients)
        
        # Fill any missing roles with the default client
        if client:
            for role in AgentRole:
                if role not in self._clients:
                    self._clients[role] = client
        
        # Validate: every role must have a client
        for role in AgentRole:
            if role not in self._clients:
                raise ValueError(f"No client configured for role {role.value}. "
                                 "Provide either 'client' (default for all) or "
                                 "complete 'clients' dict.")
        
        self.math_temp = 0.0
        self.prog_temp = 0.05
        self.enable_baseline_fallback_on_mas_failure = True
        self.enable_metamorphic_testing = False
        self.enable_hypothesis_testing = True
        
        # [v7.3] Log the configuration
        self._log_model_config()
    
    def _get_client(self, role: AgentRole) -> UnifiedLLMClient:
        """Get the client assigned to a specific agent role."""
        return self._clients[role]
    
    def _log_model_config(self):
        """Log which model is assigned to each role."""
        logger.info("=" * 50)
        logger.info("HETEROGENEOUS MODEL CONFIGURATION:")
        is_homogeneous = len(set(
            f"{c.provider}/{c.model_name}" for c in self._clients.values()
        )) == 1
        if is_homogeneous:
            c = list(self._clients.values())[0]
            logger.info(f"  [Homogeneous] All roles → {c.provider}/{c.model_name}")
        else:
            for role in AgentRole:
                c = self._clients[role]
                logger.info(f"  {role.value:<25} → {c.provider}/{c.model_name}")
        logger.info("=" * 50)
    
    def get_model_config_summary(self) -> Dict[str, str]:
        """Return a summary dict for logging/CSV output."""
        return {
            f"model_{role.value}": f"{self._clients[role].provider}/{self._clients[role].model_name}"
            for role in AgentRole
        }

    # -------------------------------------------------------------------------
    # Extract Answer (with error guard)
    # -------------------------------------------------------------------------
    
    def extract_answer(self, text: Any) -> Tuple[str, Any]:
        """
        Enhanced answer extraction with error response guard.
        """
        # [FIX v7.1] Check for error response FIRST
        if _is_error_response(text):
            logger.warning(f"extract_answer received error response: {str(text)[:150]}")
            return "unknown", None
        
        text = str(text)
        
        # Strategy 1: ANSWER: [[...]] tag
        match = re.search(r'ANSWER:\s*\[\[([^\]]+)\]\]', text, re.IGNORECASE)
        if match:
            val = match.group(1).strip()
            return val, val
        
        # Strategy 2: Extract last number
        num = _extract_last_number(text)
        if num is not None:
            return str(num), num
        
        # Strategy 3: Check for common answer patterns
        answer_patterns = [
            r'(?:answer|result|solution)\s*(?:is|=|:)\s*([0-9,.]+)',
            r'final\s+(?:answer|result)\s*(?:is|=|:)\s*([0-9,.]+)',
        ]
        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    val = float(match.group(1).replace(',', ''))
                    return str(val), val
                except:
                    pass
        
        return "unknown", None

    def _last_number_from_text(self, s: str) -> Optional[float]:
        return _extract_last_number(s)

    # -------------------------------------------------------------------------
    # Mathematician Agent (Architect)
    # -------------------------------------------------------------------------
    
    def run_mathematician_analysis(self, problem: str) -> dict:
        """
        [v9.0] Enhanced Mathematician with:
        1. Self-verification: asks the LLM to mentally compute the answer  
           from its own equations and check if it's reasonable
        2. Retry on failure: if JSON parsing fails, retries once with 
           a simpler prompt instead of returning empty blueprint
        """
        
        sys_msg = """You are an expert Mathematician analyzing word problems.

Your task: Break down the problem into a structured solution plan.

OUTPUT FORMAT (strict JSON):
{
  "unknown": "what we need to find (one sentence)",
  "givens": {
    "variable_name_1": numeric_value,
    "variable_name_2": numeric_value
  },
  "solution_steps": [
    "Step 1: Clear description of first calculation",
    "Step 2: What to calculate next using Step 1 result",
    "Step 3: Final calculation to get the answer"
  ],
  "equations": [
    "step1_result = givens['variable_name_1'] + givens['variable_name_2']",
    "step2_result = step1_result * 2",
    "answer = step2_result"
  ],
  "expected_answer": "your mental estimate of what the numeric answer should be",
  "distractor_check": "List any numbers/info in the problem to IGNORE (if any)"
}

CRITICAL RULES:
1. Extract ONLY relevant numbers into 'givens'. Ignore irrelevant numbers.
2. Use descriptive variable names (e.g., 'initial_apples', 'eaten_apples')
3. Each equation must be valid Python code referencing givens['key']
4. The LAST equation must assign to 'answer'
5. SELF-CHECK: Before outputting, mentally trace through your equations with the actual numbers. Does the result match your expected_answer? If not, fix your equations.
6. Return ONLY valid JSON, no preamble or explanation

EXAMPLE:
Problem: "Jane has 10 apples. She eats 3 and buys 5 more. How many does she have?"
Output:
{
  "unknown": "total apples Jane has",
  "givens": {"initial_apples": 10, "eaten_apples": 3, "bought_apples": 5},
  "solution_steps": [
    "Step 1: Subtract eaten from initial: 10 - 3 = 7",
    "Step 2: Add bought: 7 + 5 = 12"
  ],
  "equations": [
    "remaining = givens['initial_apples'] - givens['eaten_apples']",
    "answer = remaining + givens['bought_apples']"
  ],
  "expected_answer": "12",
  "distractor_check": "None"
}
"""
        
        msgs = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": f"Problem:\n{problem}\n\nAnalyze and return the JSON blueprint."}
        ]
        
        res = self._get_client(AgentRole.MATHEMATICIAN).call_model(
            msgs, temperature=self.math_temp, max_tokens=800
        )
        blueprint = _extract_blueprint_json(str(res))
        
        # [v9.0] If blueprint is empty (JSON parse failed), retry with simpler prompt
        if not blueprint.get("equations") and not blueprint.get("givens"):
            logger.info("Blueprint empty — retrying Mathematician with simplified prompt")
            retry_msg = f"""Solve this math problem step by step. Extract the numbers, write Python equations, and give the answer.

Problem: {problem}

Reply with ONLY this JSON (no other text):
{{"givens": {{"name": number}}, "equations": ["answer = ..."], "unknown": "what to find", "solution_steps": ["Step 1: ..."], "expected_answer": "number", "distractor_check": "None"}}"""
            
            res2 = self._get_client(AgentRole.MATHEMATICIAN).call_model(
                [{"role": "user", "content": retry_msg}],
                temperature=0.0, max_tokens=600
            )
            blueprint2 = _extract_blueprint_json(str(res2))
            if blueprint2.get("equations") or blueprint2.get("givens"):
                logger.info("Retry succeeded — got valid blueprint")
                blueprint = blueprint2
        
        return blueprint

    # -------------------------------------------------------------------------
    # Programmer Agent (Engineer)
    # -------------------------------------------------------------------------
    
    def run_programmer_solver(self, problem: str, blueprint: dict, max_attempts: int = 3) -> AgentResponse:
        
        givens = blueprint.get("givens", {})
        equations = blueprint.get("equations", [])
        solution_steps = blueprint.get("solution_steps", [])
        unknown = blueprint.get("unknown", "the answer")
        
        # [FIX v7.1] If blueprint has no equations (e.g., from error), fail fast
        if not equations and not givens:
            logger.warning("Programmer received empty blueprint (likely from API error)")
            return AgentResponse(
                agent="Programmer (empty blueprint)",
                answer="unknown",
                parsed="unknown",
                confidence=0.0,
                reasoning_trace="Blueprint was empty — likely API error",
                quality_metrics={"error": "empty_blueprint"}
            )
        
        blueprint_text = _format_blueprint_for_programmer(blueprint)
        
        sys_msg = """You are an expert Python programmer solving math problems.

STRICT RULES:
1. Start with: givens = <the exact dict from the blueprint>
2. Implement EACH equation from the blueprint IN ORDER
3. Store the final result in a variable called 'answer'
4. Print ONLY the final numeric answer (no explanations, no units)
5. Use the exact variable names from the blueprint

EXAMPLE:
Given blueprint equations:
  remaining = givens['initial'] - givens['used']
  answer = remaining

Your code:
```python
givens = {"initial": 10, "used": 3}
remaining = givens['initial'] - givens['used']
answer = remaining
print(answer)
```

OUTPUT FORMAT:
- Python code in ```python ... ``` block
- After code, write: ANSWER: [[<number>]]
"""
        
        user_msg = f"""ORIGINAL PROBLEM:
{problem}

ARCHITECT'S BLUEPRINT:
{blueprint_text}

Write the Python code to solve this. Follow the blueprint equations exactly.
"""
        
        repair_feedback = ""
        best_answer = None
        last_code = None
        
        for attempt in range(max_attempts):
            msgs = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg + repair_feedback}
            ]
            
            raw_response = self._get_client(AgentRole.PROGRAMMER).call_model(
                msgs, 
                temperature=self.prog_temp, 
                max_tokens=1000
            )
            
            # [FIX v7.1] Check for error response from LLM
            if _is_error_response(raw_response):
                logger.warning(f"Programmer attempt {attempt+1}: API returned error")
                repair_feedback = f"\n\n[Attempt {attempt+1}] LLM call failed. Retrying..."
                continue
            
            # Extract code
            code = _extract_code_from_response(str(raw_response))
            if not code:
                repair_feedback = f"\n\n[Attempt {attempt+1}] ERROR: No code block found. Use ```python ... ```"
                continue
            
            last_code = code
            
            # Execute code
            success, output = PythonExecutor.execute(code)
            if not success:
                repair_feedback = f"\n\n[Attempt {attempt+1}] EXECUTION ERROR:\n{output}\n\nFix the code and try again."
                continue
            
            # Extract answer
            answer_num = _extract_last_number(output)
            if answer_num is None:
                repair_feedback = f"\n\n[Attempt {attempt+1}] FORMAT ERROR: Output was '{output}'. Print only a number."
                continue
            
            best_answer = str(answer_num)
            
            # Optional: Metamorphic testing
            gate_log = "Metamorphic testing disabled"
            if self.enable_metamorphic_testing:
                tests = blueprint.get("metamorphic_tests", [])
                if tests:
                    ok_gate, gate_log = self._metamorphic_gate(code, tests)
                    if not ok_gate:
                        gate_log = f"WARNING: {gate_log}"
            
            # Success!
            return AgentResponse(
                agent="Programmer (optimized)",
                answer=best_answer,
                parsed=best_answer,
                confidence=1.0,
                reasoning_trace=code[:500],
                quality_metrics={
                    "execution_output": output,
                    "metamorphic_gate": gate_log,
                    "attempts": attempt + 1
                }
            )
        
        # Failed after all attempts — try SymPy symbolic solver as fallback
        sympy_answer = None
        sympy_trace = ""
        if SYMPY_AVAILABLE and blueprint.get("equations"):
            logger.info("Programmer failed. Attempting SymPy symbolic solver fallback...")
            sym_ok, sym_ans, sym_trace = SymbolicSolver.solve_from_blueprint(blueprint)
            sympy_trace = sym_trace
            if sym_ok:
                sym_num = _extract_last_number(sym_ans)
                if sym_num is not None:
                    sympy_answer = str(sym_num)
                    logger.info(f"SymPy fallback SUCCESS: {sympy_answer}")

        if sympy_answer:
            return AgentResponse(
                agent="SymPy (symbolic fallback)",
                answer=sympy_answer,
                parsed=sympy_answer,
                confidence=0.8,  # High confidence (correct arithmetic) but no code verification
                reasoning_trace=sympy_trace[:500],
                quality_metrics={
                    "solver": "sympy_symbolic",
                    "programmer_failed_attempts": max_attempts,
                    "last_code_error": repair_feedback[:200] if repair_feedback else "N/A"
                }
            )

        fallback_answer = best_answer if best_answer else "unknown"
        return AgentResponse(
            agent="Programmer (failed)",
            answer=fallback_answer,
            parsed=fallback_answer,
            confidence=0.2,
            reasoning_trace=last_code[:500] if last_code else "No code generated",
            quality_metrics={
                "error": "Max attempts reached",
                "last_feedback": repair_feedback,
                "sympy_attempted": SYMPY_AVAILABLE,
                "sympy_trace": sympy_trace[:200] if sympy_trace else "N/A"
            }
        )

    # -------------------------------------------------------------------------
    # [NEW v8.0] Process-Level Verification
    # -------------------------------------------------------------------------
    
    def verify_code_against_blueprint(self, problem: str, blueprint: dict,
                                       code: str, code_answer: str) -> Tuple[bool, str, float]:
        """
        [v9.0] Purely rule-based verification (0 API calls).
        
        Cross-checks:
        1. Givens consistency: code uses same values as blueprint
        2. Equation coverage: all blueprint equations have corresponding code
        3. Answer sanity: sign, magnitude, and expected_answer match
        """
        givens = blueprint.get("givens", {})
        equations = blueprint.get("equations", [])
        
        issues = []
        
        # Check 1: Givens consistency
        code_givens = _extract_givens_dict_from_code(code)
        if code_givens is not None and givens:
            for key, val in givens.items():
                if key not in code_givens:
                    issues.append(f"Missing given '{key}'")
                elif isinstance(val, (int, float)) and isinstance(code_givens.get(key), (int, float)):
                    if abs(code_givens[key] - val) > 1e-6:
                        issues.append(f"Givens mismatch '{key}': blueprint={val} code={code_givens[key]}")
        
        # Check 2: Equation variables in code
        for eq in equations:
            if "=" in eq:
                var_name = eq.split("=")[0].strip()
                if var_name not in code and var_name != "answer":
                    issues.append(f"Missing variable '{var_name}'")
        
        # Check 3: Answer sanity
        answer_num = _extract_last_number(code_answer)
        if answer_num is not None:
            if answer_num < 0 and not any(
                kw in problem.lower() for kw in ["loss", "decrease", "debt", "negative", "below", "fewer", "less", "owe"]
            ):
                issues.append(f"Negative answer ({answer_num}) seems wrong for this problem")
            
            if givens:
                max_given = max((abs(v) for v in givens.values() if isinstance(v, (int, float))), default=0)
                if max_given > 0 and abs(answer_num) > max_given * 10000:
                    issues.append(f"Answer ({answer_num}) implausibly large vs givens (max={max_given})")
        
        # Check 4: [v9.0] Cross-check with Mathematician's expected_answer
        expected = blueprint.get("expected_answer", "")
        if expected and answer_num is not None:
            expected_num = _extract_last_number(str(expected))
            if expected_num is not None and abs(expected_num) > 0.01:
                rel_diff = abs(answer_num - expected_num) / max(abs(expected_num), 1e-9)
                if rel_diff > 0.1:  # More than 10% off from Mathematician's estimate
                    issues.append(f"Code answer ({answer_num}) differs from Mathematician estimate ({expected_num}) by {rel_diff:.0%}")
        
        # Score
        if not issues:
            return True, "All checks passed", 1.0
        
        critical = sum(1 for i in issues if "mismatch" in i.lower() or "negative" in i.lower() or "implausibly" in i.lower())
        minor = len(issues) - critical
        
        if critical > 0:
            confidence = max(0.3, 1.0 - critical * 0.25 - minor * 0.1)
            return False, "; ".join(issues[:3]), confidence
        
        confidence = max(0.6, 1.0 - minor * 0.1)
        return True, "; ".join(issues[:3]), confidence

    # -------------------------------------------------------------------------
    # Metamorphic Testing (Optional)
    # -------------------------------------------------------------------------
    
    def _metamorphic_gate(self, code_block: str, tests: list) -> Tuple[bool, str]:
        base_givens = _extract_givens_dict_from_code(code_block)
        if base_givens is None:
            return False, "No givens dict found"
        
        ok, base_out = PythonExecutor.execute(code_block)
        if not ok:
            return False, f"Base execution failed: {base_out}"
        
        base_val = _extract_last_number(base_out)
        if base_val is None:
            return False, f"Base output not numeric: {base_out}"
        
        logs = []
        for test in tests[:3]:
            name = test.get("name", "unnamed")
            muts = test.get("mutations", [])
            assertion = test.get("assert", {})
            
            mutated_givens = dict(base_givens)
            try:
                for mu in muts:
                    var = mu["var"]
                    op = mu["op"]
                    val = mu["value"]
                    
                    if var not in mutated_givens:
                        raise KeyError(f"Variable '{var}' not in givens")
                    
                    if op == "add":
                        mutated_givens[var] += val
                    elif op == "mul":
                        mutated_givens[var] *= val
                    else:
                        raise ValueError(f"Unknown op: {op}")
            except Exception as e:
                logs.append(f"[{name}] SKIP: {e}")
                continue
            
            mutated_code = _replace_givens_dict_in_code(code_block, mutated_givens)
            ok2, out2 = PythonExecutor.execute(mutated_code)
            if not ok2:
                logs.append(f"[{name}] SKIP: Execution failed")
                continue
            
            val2 = _extract_last_number(out2)
            if val2 is None:
                logs.append(f"[{name}] SKIP: Output not numeric")
                continue
            
            atype = assertion.get("type")
            aval = assertion.get("value")
            
            passed = False
            try:
                if atype == "delta":
                    passed = abs((val2 - base_val) - float(aval)) < 1e-6
                elif atype == "ratio":
                    if abs(base_val) > 1e-6:
                        passed = abs((val2 / base_val) - float(aval)) < 1e-4
                elif atype == "monotonic":
                    if aval == "increase":
                        passed = val2 > base_val
                    elif aval == "decrease":
                        passed = val2 < base_val
            except Exception as e:
                logs.append(f"[{name}] SKIP: Assertion error - {e}")
                continue
            
            status = "PASS" if passed else "FAIL"
            logs.append(f"[{name}] {status}: base={base_val}, mutated={val2}")
            
            if not passed:
                return False, "\n".join(logs)
        
        return True, "\n".join(logs) if logs else "No tests evaluated"

    # =========================================================================
    # STRUCTURED HYPOTHESIS TESTING (SHT)
    # =========================================================================

    def _confidence_gate(self, primary_answer: str, baseline_answer: str,
                         programmer_response: AgentResponse,
                         blueprint: dict) -> Tuple[bool, str]:
        # Criterion 1: Programmer failed entirely
        if str(primary_answer).strip().lower() == "unknown":
            return False, "programmer_failed"

        # Criterion 2: Primary answer disagrees with baseline
        primary_num = _extract_last_number(primary_answer)
        baseline_num = _extract_last_number(baseline_answer)
        if primary_num is not None and baseline_num is not None:
            if abs(primary_num - baseline_num) > 1e-3:
                return False, "baseline_disagreement"
        elif str(primary_answer).strip() != str(baseline_answer).strip():
            return False, "baseline_disagreement"

        # Criterion 3: Programmer exhausted all repair attempts
        if programmer_response.quality_metrics.get("error") == "Max attempts reached":
            return False, "max_attempts_exhausted"

        # Criterion 4: Sanity checks
        if primary_num is not None:
            if primary_num < 0:
                return False, "negative_answer"

            givens = blueprint.get("givens", {})
            if givens:
                max_given = max(
                    (abs(v) for v in givens.values() if isinstance(v, (int, float))),
                    default=0
                )
                if max_given > 0 and abs(primary_num) > max_given * 10000:
                    return False, "answer_magnitude_suspicious"

        # [FIX v7.1] Criterion 5: Both answers are "unknown" (total API failure)
        if str(baseline_answer).strip().lower() == "unknown":
            return False, "baseline_also_failed"

        return True, "all_checks_passed"

    STRATEGY_ARCHETYPES = [
        "Arithmetic-Sequential: chain of basic operations (add, subtract, multiply, divide) applied step by step",
        "Algebraic-Equational: set up one or more equations with unknowns and solve symbolically",
        "Unit-Rate: compute a per-unit rate first, then scale to the target quantity",
        "Working-Backwards: start from what the answer should look like and reverse the operations",
        "Partitioning: split the problem into independent sub-problems, solve each, then combine",
    ]

    def generate_alternative_hypotheses(self, problem: str,
                                        primary_blueprint: dict,
                                        primary_answer: str) -> List[dict]:
        """
        [v9.0] Critic-based hypothesis generation.
        
        Instead of "generate 2 completely different approaches" (which produces
        correlated errors from the same model), we now ask:
        
        1. CRITIC: "Review this solution. What errors do you find?"
        2. CORRECTION: "Provide a corrected solution fixing those errors."
        
        This transforms SHT from parallel exploration into peer review,
        which is fundamentally more useful for catching reasoning errors.
        """
        primary_eqs = primary_blueprint.get("equations", [])
        primary_givens = primary_blueprint.get("givens", {})
        primary_steps = primary_blueprint.get("solution_steps", [])

        sys_msg = f"""You are a meticulous Mathematics Reviewer. 
Your job is to FIND ERRORS in a proposed solution and provide CORRECTIONS.

A colleague solved a math problem and got the answer: {primary_answer}

REVIEW CHECKLIST:
1. Are all relevant numbers from the problem extracted correctly?
2. Are any IRRELEVANT numbers (distractors) mistakenly included?
3. Is each mathematical operation correct for what the problem asks?
4. Are there any MISSING steps?
5. Does the final answer actually answer what was asked?

After your review, provide exactly 2 corrected solutions:
- Correction 1: Fix the most likely error you found
- Correction 2: Solve from scratch using a completely different approach

OUTPUT FORMAT (strict JSON, no other text):
{{
  "review": "Brief description of error(s) found (or 'no errors found')",
  "alternatives": [
    {{
      "strategy_name": "correction_of_[specific error]",
      "error_found": "what was wrong in the original",
      "unknown": "what we need to find",
      "givens": {{"var_name": numeric_value, ...}},
      "solution_steps": ["Step 1: ...", "Step 2: ..."],
      "equations": ["step1 = givens['var'] ...", "answer = ..."],
      "expected_answer": "your mental estimate"
    }},
    {{
      "strategy_name": "independent_rederivation",
      "error_found": "solving from scratch to verify",
      "unknown": "what we need to find",
      "givens": {{"var_name": numeric_value, ...}},
      "solution_steps": ["Step 1: ...", "Step 2: ..."],
      "equations": ["step1 = givens['var'] ...", "answer = ..."],
      "expected_answer": "your mental estimate"
    }}
  ]
}}"""

        user_msg = f"""PROBLEM:
{problem}

COLLEAGUE'S SOLUTION TO REVIEW:
Givens: {json.dumps(primary_givens)}
Steps: {json.dumps(primary_steps)}
Equations: {json.dumps(primary_eqs)}
Answer obtained: {primary_answer}

Review for errors and provide 2 corrected/alternative solutions as JSON."""

        msgs = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}
        ]

        raw = self._get_client(AgentRole.HYPOTHESIS_GENERATOR).call_model(
            msgs, temperature=0.3, max_tokens=1200
        )

        if _is_error_response(raw):
            logger.warning("Hypothesis generator returned error response")
            return []

        alternatives = []
        try:
            text = str(raw).strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()

            parsed = None
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                start, end = text.find("{"), text.rfind("}")
                if start != -1 and end > start:
                    try:
                        parsed = json.loads(text[start:end+1])
                    except json.JSONDecodeError:
                        pass

            if parsed and "alternatives" in parsed:
                review = parsed.get("review", "")
                if review:
                    logger.info(f"SHT Critic review: {review[:150]}")
                    
                for alt in parsed["alternatives"][:2]:
                    if isinstance(alt, dict):
                        alt.setdefault("unknown", "the answer")
                        alt.setdefault("givens", {})
                        alt.setdefault("solution_steps", [])
                        alt.setdefault("equations", [])
                        alt.setdefault("strategy_name", "critic_correction")
                        alternatives.append(alt)
        except Exception as e:
            logger.warning(f"SHT: Failed to parse critic response: {e}")

        return alternatives

    def _triage_candidates(self, candidates: List[HypothesisResult]) -> Tuple[Optional[str], Optional[str], str]:
        valid = [c for c in candidates if c.code_success and c.parsed_answer is not None]

        if not valid:
            return None, None, "no_valid_candidates"

        groups: Dict[str, List[HypothesisResult]] = {}
        for c in valid:
            matched = False
            for key in groups:
                if abs(c.parsed_answer - float(key)) < 1e-3:
                    groups[key].append(c)
                    matched = True
                    break
            if not matched:
                groups[str(c.parsed_answer)] = [c]

        if not groups:
            return None, None, "no_valid_candidates"

        sorted_groups = sorted(
            groups.items(),
            key=lambda g: (len(g[1]), sum(c.confidence for c in g[1]) / len(g[1])),
            reverse=True
        )

        best_answer_key, best_group = sorted_groups[0]

        if len(sorted_groups) == 1:
            winner = best_group[0]
            return winner.answer, winner.strategy_name, "unanimous"

        if len(best_group) >= 2 and (len(sorted_groups) < 2 or len(best_group) > len(sorted_groups[1][1])):
            winner = best_group[0]
            return winner.answer, winner.strategy_name, "majority"

        return None, None, "no_majority"

    def _judge_hypotheses(self, problem: str,
                          candidates: List[HypothesisResult]) -> Tuple[str, str, str]:
        candidate_summaries = []
        for i, c in enumerate(candidates):
            if not c.code_success:
                status = f"FAILED (error: {c.execution_output[:100]})"
            else:
                status = f"SUCCESS → answer = {c.answer}"

            summary = f"""--- Candidate {i+1}: {c.strategy_name} ({c.hypothesis_id}) ---
Status: {status}
Equations: {json.dumps(c.blueprint.get('equations', []))}
Code (first 300 chars): {(c.code or 'N/A')[:300]}
"""
            candidate_summaries.append(summary)

        sys_msg = """You are a mathematical reasoning Judge. Multiple solution strategies were tried for the same problem. Some may have errors.

Your task: Evaluate each candidate's reasoning and select the MOST RELIABLE answer.

Evaluation criteria (in order of importance):
1. CODE EXECUTION: Did the code run successfully? Discard failed candidates.
2. MATHEMATICAL CORRECTNESS: Are the equations and logic sound?
3. COMPLETENESS: Does the approach account for ALL conditions in the problem?
4. AGREEMENT: If multiple strategies agree on an answer, that's strong evidence.
5. SIMPLICITY: Among equally valid approaches, prefer the simpler one.

OUTPUT FORMAT:
First explain your reasoning briefly (2-3 sentences).
Then write: SELECTED_ANSWER: [[number]]
Then write: SELECTED_STRATEGY: [[strategy_name]]"""

        user_msg = f"""PROBLEM:
{problem}

CANDIDATES:
{''.join(candidate_summaries)}

Evaluate and select the most reliable answer."""

        msgs = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}
        ]

        raw = self._get_client(AgentRole.JUDGE).call_model(msgs, temperature=0.0, max_tokens=800)
        
        # [FIX v7.1] Check for error response from judge
        if _is_error_response(raw):
            logger.warning("Judge returned error response")
            return "unknown", "judge_error", "Judge API call failed"
        
        raw_text = str(raw)

        answer_match = re.search(r'SELECTED_ANSWER:\s*\[\[([^\]]+)\]\]', raw_text)
        strategy_match = re.search(r'SELECTED_STRATEGY:\s*\[\[([^\]]+)\]\]', raw_text)

        if answer_match:
            judge_answer = answer_match.group(1).strip()
        else:
            num = _extract_last_number(raw_text)
            judge_answer = str(num) if num is not None else "unknown"

        judge_strategy = strategy_match.group(1).strip() if strategy_match else "judge_selection"

        return judge_answer, judge_strategy, raw_text[:500]

    def _structured_hypothesis_testing(self, problem: str, expected: str,
                                       primary_blueprint: dict,
                                       programmer_response: AgentResponse,
                                       baseline_answer: str) -> HypothesisLog:
        primary_answer = programmer_response.answer
        primary_num = _extract_last_number(primary_answer)

        log = HypothesisLog(
            problem=problem,
            expected=expected,
            final_answer=primary_answer,
            final_strategy="primary",
        )

        primary_candidate = HypothesisResult(
            hypothesis_id="primary",
            strategy_name="primary_blueprint",
            blueprint=primary_blueprint,
            code=programmer_response.reasoning_trace,
            code_success=programmer_response.confidence > 0.5,
            execution_output=programmer_response.quality_metrics.get("execution_output", ""),
            answer=primary_answer,
            parsed_answer=primary_num,
            confidence=programmer_response.confidence,
            agent_response=programmer_response,
        )
        log.candidates.append(primary_candidate)

        baseline_num = _extract_last_number(baseline_answer)
        baseline_candidate = HypothesisResult(
            hypothesis_id="baseline",
            strategy_name="zero_shot_baseline",
            blueprint={},
            code=None,
            code_success=baseline_num is not None,
            execution_output="",
            answer=baseline_answer,
            parsed_answer=baseline_num,
            confidence=0.5,
            agent_response=None,
        )
        log.candidates.append(baseline_candidate)

        is_confident, gate_reason = self._confidence_gate(
            primary_answer, baseline_answer, programmer_response, primary_blueprint
        )

        if is_confident:
            log.triage_result = "confident_skip"
            log.final_answer = primary_answer
            log.final_strategy = "primary_blueprint"
            log.hypothesis_testing_triggered = False
            log.api_calls_used = 3
            return log

        logger.info(f"SHT triggered: {gate_reason}")
        log.hypothesis_testing_triggered = True
        api_calls = 3

        # [FIX v7.2] Check if we can afford SHT calls (~4 more calls × 1500 tokens)
        sht_cost_estimate = 4 * 1500  # hypothesis gen + 2 programmer + maybe judge
        if not token_budget.can_afford(sht_cost_estimate):
            logger.warning("SHT skipped due to token budget. Using best available answer.")
            # Fall back to whichever of primary/baseline looks better
            if primary_num is not None:
                log.final_answer = primary_answer
                log.final_strategy = "primary_budget_skip"
            else:
                log.final_answer = baseline_answer
                log.final_strategy = "baseline_budget_skip"
            log.triage_result = "budget_skip"
            log.api_calls_used = 3
            return log

        alt_blueprints = self.generate_alternative_hypotheses(
            problem, primary_blueprint, primary_answer
        )
        api_calls += 1

        for idx, alt_bp in enumerate(alt_blueprints[:2]):
            alt_response = self.run_programmer_solver(problem, alt_bp, max_attempts=1)
            api_calls += 1

            alt_num = _extract_last_number(alt_response.answer)
            alt_candidate = HypothesisResult(
                hypothesis_id=f"alt_{idx+1}",
                strategy_name=alt_bp.get("strategy_name", f"alternative_{idx+1}"),
                blueprint=alt_bp,
                code=alt_response.reasoning_trace,
                code_success=alt_response.confidence > 0.5,
                execution_output=alt_response.quality_metrics.get("execution_output", ""),
                answer=alt_response.answer,
                parsed_answer=alt_num,
                confidence=alt_response.confidence,
                agent_response=alt_response,
            )
            log.candidates.append(alt_candidate)

        triage_answer, triage_strategy, triage_method = self._triage_candidates(log.candidates)

        if triage_method in ("unanimous", "majority"):
            log.triage_result = triage_method
            log.final_answer = triage_answer
            log.final_strategy = triage_strategy
            log.api_calls_used = api_calls
            return log

        judge_answer, judge_strategy, judge_reasoning = self._judge_hypotheses(
            problem, log.candidates
        )
        api_calls += 1

        log.triage_result = "judge"
        log.judge_reasoning = judge_reasoning
        log.final_answer = judge_answer
        log.final_strategy = judge_strategy
        log.api_calls_used = api_calls
        return log

    # -------------------------------------------------------------------------
    # Main Solve Method
    # -------------------------------------------------------------------------

    def solve(self, problem: str, expected: str) -> Dict[str, Any]:
        # Step 1: Baseline
        baseline_prompt = f"{problem}\n\nSolve this step-by-step. End with: ANSWER: [[numeric_value]]"
        base_raw = self._get_client(AgentRole.BASELINE).call_model(
            [{"role": "user", "content": baseline_prompt}],
            temperature=0.1,
            max_tokens=800
        )
        base_ans, _ = self.extract_answer(base_raw)

        # Step 2: Architect
        blackboard_logic = self.run_mathematician_analysis(problem)

        # Step 3: Engineer (with SymPy fallback built-in)
        programmer_response = self.run_programmer_solver(problem, blackboard_logic)

        # Step 3b: [NEW v8.0] Process-Level Verification
        verification_passed = True
        verification_feedback = "Skipped"
        verification_confidence = 1.0
        
        if (programmer_response.confidence > 0.5
            and programmer_response.answer != "unknown"
            and programmer_response.reasoning_trace
            and "SymPy" not in programmer_response.agent):
            # Only verify code-based solutions, not SymPy fallbacks
            verification_passed, verification_feedback, verification_confidence = \
                self.verify_code_against_blueprint(
                    problem, blackboard_logic,
                    programmer_response.reasoning_trace,
                    programmer_response.answer
                )
            
            # Adjust programmer confidence based on verification
            adjusted_confidence = programmer_response.confidence * verification_confidence
            programmer_response = AgentResponse(
                agent=programmer_response.agent,
                answer=programmer_response.answer,
                parsed=programmer_response.parsed,
                confidence=adjusted_confidence,
                reasoning_trace=programmer_response.reasoning_trace,
                quality_metrics={
                    **programmer_response.quality_metrics,
                    "verification_passed": verification_passed,
                    "verification_confidence": verification_confidence,
                    "verification_feedback": verification_feedback[:200],
                }
            )
            
            if not verification_passed:
                logger.info(f"Process verification FAILED (conf={verification_confidence:.2f}). "
                            f"Trying SymPy as alternative...")
                if SYMPY_AVAILABLE and blackboard_logic.get("equations"):
                    sym_ok, sym_ans, sym_trace = SymbolicSolver.solve_from_blueprint(blackboard_logic)
                    if sym_ok:
                        sym_num = _extract_last_number(sym_ans)
                        if sym_num is not None:
                            programmer_response = AgentResponse(
                                agent="SymPy (post-verification fallback)",
                                answer=str(sym_num),
                                parsed=str(sym_num),
                                confidence=0.75,
                                reasoning_trace=sym_trace[:500],
                                quality_metrics={
                                    "solver": "sympy_post_verification",
                                    "original_answer": programmer_response.answer,
                                    "verification_rejection": verification_feedback[:200],
                                }
                            )

        # Step 4: Structured Hypothesis Testing
        hypothesis_log = None
        if self.enable_hypothesis_testing:
            hypothesis_log = self._structured_hypothesis_testing(
                problem, expected, blackboard_logic,
                programmer_response, base_ans
            )
            mas_answer = hypothesis_log.final_answer
            used_baseline_fallback = False
        else:
            mas_answer = programmer_response.answer
            used_baseline_fallback = False

        # Step 5: Fallback
        if self.enable_baseline_fallback_on_mas_failure:
            if str(mas_answer).strip().lower() == "unknown" and str(base_ans).strip().lower() != "unknown":
                mas_answer = base_ans
                used_baseline_fallback = True

        result = {
            "problem": problem,
            "expected": expected,
            "baseline": {
                "answer": base_ans,
                "model": str(self._get_client(AgentRole.BASELINE)),  # [v7.3]
            },
            "mas": {
                "answer": mas_answer,
                "logic_trace": json.dumps(blackboard_logic, ensure_ascii=False)[:500],
                "used_baseline_fallback": used_baseline_fallback,
                "programmer_metrics": programmer_response.quality_metrics,
                "verification": {
                    "passed": verification_passed,
                    "confidence": verification_confidence,
                    "feedback": verification_feedback[:200],
                },
            },
            "agents": [programmer_response],
            "model_config": self.get_model_config_summary(),  # [v7.3]
        }

        if hypothesis_log:
            result["sht"] = {
                "triggered": hypothesis_log.hypothesis_testing_triggered,
                "triage_result": hypothesis_log.triage_result,
                "final_strategy": hypothesis_log.final_strategy,
                "num_candidates": len(hypothesis_log.candidates),
                "api_calls_used": hypothesis_log.api_calls_used,
                "judge_reasoning": hypothesis_log.judge_reasoning,
                "candidates": [
                    {
                        "id": c.hypothesis_id,
                        "strategy": c.strategy_name,
                        "answer": c.answer,
                        "success": c.code_success,
                    }
                    for c in hypothesis_log.candidates
                ],
            }

        return result


# --------------------------- Main Pipeline ---------------------------

class QualityAwarePipeline:
    def __init__(self, provider: str = "groq", use_cache: bool = False,
                 heterogeneous_preset: Optional[str] = None,
                 custom_config: Optional[Dict[AgentRole, ModelConfig]] = None):
        """
        [UPDATED v7.3] Supports heterogeneous model configuration.
        
        Args:
            provider: Default provider (used if no heterogeneous config).
            use_cache: Whether to cache API calls.
            heterogeneous_preset: Name of a preset from HETEROGENEOUS_PRESETS.
            custom_config: Custom Dict[AgentRole, ModelConfig] mapping.
        """
        self.manager = EnhancedProblemManager(random_seed=None)
        self.results: List[Dict[str, Any]] = []
        
        # Determine model configuration
        if custom_config:
            role_config = custom_config
        elif heterogeneous_preset and heterogeneous_preset in HETEROGENEOUS_PRESETS:
            role_config = HETEROGENEOUS_PRESETS[heterogeneous_preset]
        else:
            # Backward compatible: single provider for all roles
            role_config = {
                role: ModelConfig(provider, None)
                for role in AgentRole
            }
        
        # [v7.3] Build one LLMClient per unique (provider, model_name) pair
        # This avoids creating duplicate clients for the same model
        self._client_cache: Dict[str, UnifiedLLMClient] = {}
        clients: Dict[AgentRole, UnifiedLLMClient] = {}
        
        for role, mc in role_config.items():
            cache_key = f"{mc.provider}:{mc.model_name or 'default'}"
            if cache_key not in self._client_cache:
                self._client_cache[cache_key] = UnifiedLLMClient(
                    provider=mc.provider,
                    use_cache=use_cache,
                    model_override=mc.model_name
                )
            clients[role] = self._client_cache[cache_key]
        
        # Store primary client for validation
        self.client = clients[AgentRole.MATHEMATICIAN]  # Use mathematician for validation
        
        # Create solver with heterogeneous clients
        self.solver = QualityEnhancedMultiAgentSolver(clients=clients)

    def _extract_gold_answer(self, text: Any) -> Optional[float]:
        text = str(text)
        if "####" in text:
            raw_gold = text.split("####")[-1].strip()
        else:
            raw_gold = text
        nums = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", raw_gold)
        if nums:
            try:
                return float(nums[-1].replace(",", ""))
            except ValueError:
                return None
        return None

    def check_correctness(self, pred: Any, gold_text: Any) -> bool:
        gold_val = self._extract_gold_answer(gold_text)
        if gold_val is None:
            return str(pred).strip() == str(gold_text).strip()
        try:
            pred_str = str(pred).replace("$", "").replace(",", "")
            pred_nums = re.findall(r"-?\d+(?:\.\d+)?", pred_str)
            if not pred_nums:
                return False
            return abs(float(pred_nums[-1]) - gold_val) < 1e-3
        except:
            return False

    def run(self, datasets_list=["gsm8k_test"], num_problems=10, hardener: Optional[str] = None) -> pd.DataFrame:
        
        # [FIX v7.1] Validate API connection before running
        if not self.client.validate_connection():
            logger.error("="*60)
            logger.error("FATAL: API connection failed! Cannot proceed.")
            logger.error("Please check:")
            logger.error("  1. Your .env file has the correct API key")
            logger.error("  2. The API key is not expired")
            logger.error("  3. You have sufficient API credits")
            logger.error("  4. The API service is not down")
            logger.error("="*60)
            raise ConnectionError(
                "API connection validation failed. Check your API key and .env file. "
                "All results would be errors (like the 401.0 issue)."
            )
        
        logger.info(f"Pipeline started. Fetching {num_problems} random problems from: {datasets_list} | hardener={hardener}")
        problems = self.manager.load_random_problems(datasets_list, num_problems, hardener=hardener)

        detailed = []
        # [FIX v7.1] Track consecutive API errors to detect persistent failures
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 5
        
        for i, p in enumerate(problems):
            logger.info(f"Processing {i+1}/{len(problems)} (ID: {p['id']}, DS: {p['dataset']}) | {token_budget.usage_report()}")
            
            # [FIX v7.2] Check token budget before each problem
            tokens_per_problem = 9000 if self.solver.enable_hypothesis_testing else 4500
            if not token_budget.can_afford(tokens_per_problem):
                logger.error(
                    f"TOKEN BUDGET EXHAUSTED after {i} problems. "
                    f"{token_budget.usage_report()}. "
                    f"Stopping early to avoid silent failures."
                )
                break
            
            res = self.solver.solve(p["puzzle"], p["answer"])
            res["baseline"]["correct"] = self.check_correctness(res["baseline"]["answer"], p["answer"])
            res["mas"]["correct"] = self.check_correctness(res["mas"]["answer"], p["answer"])
            res["id"] = p["id"]
            res["dataset"] = p["dataset"]
            detailed.append(res)
            
            # [FIX v7.1] Check for persistent API failures
            if res["baseline"]["answer"] == "unknown" and res["mas"]["answer"] == "unknown":
                consecutive_errors += 1
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logger.error(f"ABORTING: {MAX_CONSECUTIVE_ERRORS} consecutive problems returned 'unknown'. "
                                 "API is likely down or key is invalid.")
                    break
            else:
                consecutive_errors = 0

        self.results = detailed
        sht_data = []
        for r in detailed:
            sht = r.get("sht", {})
            sht_data.append({
                "sht_triggered": sht.get("triggered", False),
                "sht_triage_result": sht.get("triage_result", "n/a"),
                "sht_winning_strategy": sht.get("final_strategy", "n/a"),
                "sht_num_candidates": sht.get("num_candidates", 0),
                "sht_api_calls": sht.get("api_calls_used", 3),
            })

        df = pd.DataFrame([
            {
                "id": r["id"],
                "dataset": r.get("dataset", ""),
                "baseline_correct": r["baseline"]["correct"],
                "mas_correct": r["mas"]["correct"],
                "baseline_ans": r["baseline"]["answer"],
                "mas_ans": r["mas"]["answer"],
                "mas_used_baseline_fallback": r["mas"].get("used_baseline_fallback", False),
                "expected_snippet": str(r["expected"])[-30:],
                # [v8.0] Verification metrics
                "verification_passed": r.get("mas", {}).get("verification", {}).get("passed", True),
                "verification_confidence": r.get("mas", {}).get("verification", {}).get("confidence", 1.0),
                # [v8.0] Solver type (Programmer, SymPy, baseline fallback)
                "solver_agent": r.get("agents", [{}])[0].agent if r.get("agents") else "unknown",
                **sht_data[i],
                **r.get("model_config", {}),
            } for i, r in enumerate(detailed)
        ])
        return df

    def report(self):
        if not self.results:
            return
        rows = []
        for r in self.results:
            sht = r.get("sht", {})
            rows.append({
                "base": 1 if r["baseline"]["correct"] else 0,
                "mas": 1 if r["mas"]["correct"] else 0,
                "mas_fallback": 1 if r["mas"].get("used_baseline_fallback", False) else 0,
                "sht_triggered": 1 if sht.get("triggered", False) else 0,
                "sht_triage": sht.get("triage_result", "n/a"),
            })
        df = pd.DataFrame(rows)
        n = len(df)
        base_acc = df["base"].mean()
        mas_acc = df["mas"].mean()
        fb_rate = df["mas_fallback"].mean()
        sht_trigger_rate = df["sht_triggered"].mean()

        rescue_count = 0
        damage_count = 0
        for r in self.results:
            sht = r.get("sht", {})
            if not sht.get("triggered", False):
                continue
            primary_candidates = [c for c in sht.get("candidates", []) if c["id"] == "primary"]
            if primary_candidates:
                primary_ans = primary_candidates[0]["answer"]
                primary_correct = self.check_correctness(primary_ans, r["expected"])
                mas_correct = r["mas"]["correct"]
                if mas_correct and not primary_correct:
                    rescue_count += 1
                elif not mas_correct and primary_correct:
                    damage_count += 1

        sht_triggered_total = int(df["sht_triggered"].sum())

        print("\n" + "="*60)
        print("   PERFORMANCE REPORT (MAS + Structured Hypothesis Testing)")
        print("="*60)
        
        # [v7.3] Show model configuration
        if self.results and "model_config" in self.results[0]:
            mc = self.results[0]["model_config"]
            is_homogeneous = len(set(mc.values())) == 1
            if is_homogeneous:
                print(f"Model Config: Homogeneous ({list(mc.values())[0]})")
            else:
                print("Model Config: HETEROGENEOUS")
                for role_key, model_str in mc.items():
                    print(f"  {role_key:<25} → {model_str}")
            print("-" * 60)
        
        print(f"Total Examples: {n}")
        print("-" * 60)
        print(f"{'Metric':<30} | {'Value':<10}")
        print("-" * 60)
        print(f"{'Baseline Accuracy':<30} | {base_acc:.2%}")
        print(f"{'MAS+SHT Accuracy':<30} | {mas_acc:.2%}")
        print(f"{'Improvement over Baseline':<30} | {(mas_acc - base_acc):+.2%}")
        print(f"{'MAS->Baseline Fallback':<30} | {fb_rate:.2%}")
        print("-" * 60)
        
        # [v8.0] Verification and SymPy stats
        verif_failed = sum(
            1 for r in self.results
            if not r.get("mas", {}).get("verification", {}).get("passed", True)
        )
        sympy_used = sum(
            1 for r in self.results
            if r.get("agents") and "SymPy" in str(r["agents"][0].agent)
        )
        print(f"{'Verification Failures':<30} | {verif_failed}/{n}")
        print(f"{'SymPy Fallback Used':<30} | {sympy_used}/{n}")
        print("-" * 60)
        print(f"{'SHT Trigger Rate':<30} | {sht_trigger_rate:.2%} ({sht_triggered_total}/{n})")
        print(f"{'SHT Rescue (fixed wrong)':<30} | {rescue_count}")
        print(f"{'SHT Damage (broke correct)':<30} | {damage_count}")
        if sht_triggered_total > 0:
            print(f"{'SHT Net Benefit':<30} | {rescue_count - damage_count:+d} problems")
            triage_counts = df[df["sht_triggered"] == 1]["sht_triage"].value_counts()
            print("-" * 60)
            print("SHT Triage Breakdown:")
            for method, count in triage_counts.items():
                print(f"  {method:<26} | {count}")
        print("="*60 + "\n")


# --------------------------- Entrypoint ---------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  Multi-Agent Math Solver - VERSION 9.0 (Critic-SHT)")
    print("  Heterogeneous Models + Process Verification + SymPy Fallback")
    print("  + Structured Hypothesis Testing (SHT)")
    print("=" * 70)
    print()
    
    print("TOKEN BUDGET (Groq Free Tier = 100K tokens/day):")
    print("  10 problems + SHT  →  ~60K-90K tokens  (safe)")
    print("  20 problems + SHT  →  ~90K-120K tokens  (may hit limit)")
    print()

    print("Select Model Configuration:")
    print("1) Homogeneous Groq      — all roles use LLaMA 3.3 70B")
    print("2) Diverse Groq           — LLaMA 70B + Gemma 9B + Mixtral 8x7B")
    print("3) Cross-Provider         — Groq (LLaMA) + Google (Gemini)")
    print("4) Budget-Optimized       — LLaMA 8B for cheap roles, 70B for critical")
    print("5) Homogeneous Google     — all roles use Gemini")
    
    config_choice = input("Enter selection (1-5) [default=1]: ").strip()
    
    preset_map = {
        "1": "homogeneous_groq",
        "2": "diverse_groq",
        "3": "cross_provider",
        "4": "budget_optimized",
        "5": "homogeneous_google",
    }
    preset_name = preset_map.get(config_choice, "homogeneous_groq")
    
    print(f"\nSelected: {preset_name}")
    selected_config = HETEROGENEOUS_PRESETS[preset_name]
    print("Role assignments:")
    for role, mc in selected_config.items():
        print(f"  {role.value:<25} → {mc.provider}/{mc.model_name or 'default'}")
    print()
    
    # Number of problems
    num_input = input("Number of problems [default=10]: ").strip()
    num_problems = int(num_input) if num_input.isdigit() else 10
    
    # SHT toggle
    sht_input = input("Enable SHT hypothesis testing? (y/n) [default=y]: ").strip().lower()
    enable_sht = sht_input != "n"
    
    # Cache ON by default
    pipeline = QualityAwarePipeline(
        use_cache=True,
        heterogeneous_preset=preset_name
    )
    
    # Configure SHT
    pipeline.solver.enable_hypothesis_testing = enable_sht
    
    estimated_tokens = num_problems * (9000 if enable_sht else 4500)
    print(f"\nEstimated token usage: ~{estimated_tokens:,} tokens")
    
    # Check if cross-provider needs both keys
    providers_needed = set(mc.provider for mc in selected_config.values())
    if "groq" in providers_needed and not GROQ_API_KEY:
        print("ERROR: This config requires GROQ_API_KEY in .env")
        exit(1)
    if "google" in providers_needed and not GOOGLE_API_KEY:
        print("ERROR: This config requires GOOGLE_API_KEY in .env")
        exit(1)
    
    print()

    df_results = pipeline.run(
        datasets_list=["gsm-plus", "gsm-symbolic-p2", "gsm-hard", "svamp", "gsm8k_test"],
        num_problems=num_problems,
        hardener="distractor",
    )
    pipeline.report()
    
    print(f"\n{token_budget.usage_report()}")
    
    out_file = f"final_results_v73_{preset_name}_n{num_problems}.csv"
    df_results.to_csv(out_file, index=False)
    print(f"Results saved to '{out_file}'.")