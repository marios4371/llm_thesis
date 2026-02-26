"""
Enhanced Reasoning Quality Evaluation System for MAS Math Solver
VERSION 7.0: Structured Hypothesis Testing (SHT)
    Architect-Engineer Pattern + Multi-Hypothesis Verification

MAJOR IMPROVEMENTS (v6.2 base):
- Structured Mathematician prompt with explicit JSON schema + example
- Clear Programmer instructions that follow blueprint exactly
- Better blueprint extraction and code parsing
- Robust answer extraction with multiple strategies
- Smart baseline fallback logic
- Lower temperatures for more deterministic code generation

NEW IN v7.0 - STRUCTURED HYPOTHESIS TESTING (SHT):
- Confidence gate detects uncertainty (baseline disagreement, sanity checks)
- Generates 2 structurally different alternative solution strategies
- Each alternative uses a different mathematical archetype
- Rule-based triage (majority voting) or LLM Judge for final selection
- Full hypothesis logging for research analysis

Expected Performance:
- Baseline: ~83% accuracy
- MAS only: ~83% accuracy
- MAS + SHT: ~87-93% accuracy (hypothesis testing on uncertain problems)
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


# --------------------------- Configuration ---------------------------

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


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

groq_limiter = RateLimiter(requests_per_minute=12)
google_limiter = RateLimiter(requests_per_minute=12)

def _safe_json_load(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None

# OPTIMIZED: Better blueprint extraction with structured fallback
def _extract_blueprint_json(text: str) -> dict:
    """
    Enhanced JSON extraction with better fallback handling.
    Ensures required keys exist even if parsing fails.
    """
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
        "notes": text[:800]  # Keep context for debugging
    }

def _extract_givens_dict_from_code(code_str: str) -> Optional[dict]:
    """
    Finds: givens = {...}
    Returns dict via ast.literal_eval (safer than eval).
    """
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

# --------------------------- OPTIMIZED: Robust Code Extractor ---------------------------

def _extract_code_from_response(raw: str) -> Optional[str]:
    """
    Enhanced code extraction with multiple pattern matching strategies.
    Handles: fenced blocks, open fences, and bare code starting with 'givens ='.
    """
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
            # Stop at common delimiters
            code = re.split(r"\n\n(?:ANSWER|---|Note|Explanation)", code, maxsplit=1)[0]
            return code.strip()
    
    # Strategy 3: Code starts with "givens = " (no fence at all)
    givens_match = re.search(r"^(givens\s*=\s*\{.*)", s, re.DOTALL | re.MULTILINE)
    if givens_match:
        code = givens_match.group(1)
        # Stop at ANSWER tag or blank lines
        code = re.split(r"\n\n(?:ANSWER|---)", code, maxsplit=1)[0]
        return code.strip()
    
    return None


# OPTIMIZED: Better number extraction
def _extract_last_number(text: str) -> Optional[float]:
    """
    Extract the last numeric value from text, handling various formats.
    """
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


# OPTIMIZED: Structured blueprint formatting for Programmer
def _format_blueprint_for_programmer(bp: dict) -> str:
    """
    Format blueprint in a clear, structured way for the Programmer to follow.
    Preserves all critical information without lossy compaction.
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
        # Security checks
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
            
            # If no print output, check for 'answer' variable
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

# --------------------------- Unified LLM Client ---------------------------

class UnifiedLLMClient:
    def __init__(self, provider: str = "groq", use_cache: bool = False):
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
            if not GROQ_API_KEY: raise ValueError("Missing GROQ_API_KEY")
            self.client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
            self.model_name = "llama-3.3-70b-versatile"
        elif provider == "google":
            if not GOOGLE_API_KEY: raise ValueError("Missing GOOGLE_API_KEY")
            if not GOOGLE_AVAILABLE: raise ImportError("Google SDK missing.")
            genai.configure(api_key=GOOGLE_API_KEY)
            self._setup_google_model()
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def call_model(self, messages: List[Dict[str, str]], temperature: float = 0.3, max_tokens: int = 1200) -> Any:
        key = _make_cache_key(self.provider, self.model_name, messages, temperature)

        if self.use_cache and key in CALL_CACHE:
            return CALL_CACHE[key]

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
                return getattr(resp, "text", "")

        for attempt in range(6):
            try:
                res = _call_once()
                if self.use_cache:
                    CALL_CACHE[key] = res
                    with open(CACHE_FILE, "wb") as f:
                        pickle.dump(CALL_CACHE, f)
                return res
            except Exception as e:
                last_err = str(e)
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
        """
        Hardener should NOT change the correct answer.
        Goal: make reading/comprehension harder (like humans get distracted by irrelevant info).
        """
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


# --------------------------- OPTIMIZED Solver (Architect-Engineer Pattern) ---------------------------

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
    hypothesis_id: str          # "primary", "alt_1", "alt_2", "baseline"
    strategy_name: str          # "arithmetic_sequential", "algebraic", etc.
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
    triage_result: Optional[str] = None      # "confident_skip", "majority", "judge"
    judge_reasoning: Optional[str] = None
    final_answer: str = "unknown"
    final_strategy: str = "none"
    hypothesis_testing_triggered: bool = False
    api_calls_used: int = 3

class QualityEnhancedMultiAgentSolver:
    """
    OPTIMIZED Multi-Agent Solver with improved prompts and robustness.
    
    Key changes from original:
    - Clearer, more structured Mathematician prompt with JSON schema + example
    - Explicit Programmer instructions that follow blueprint step-by-step
    - Better error handling and answer extraction
    - Smarter baseline fallback
    - Lower temperatures for more deterministic generation
    """
    
    def __init__(self, client: UnifiedLLMClient):
        self.client = client
        self.math_temp = 0.0    # Deterministic for Mathematician
        self.prog_temp = 0.05   # OPTIMIZED: Lower temperature for more deterministic code
        self.enable_baseline_fallback_on_mas_failure = True
        self.enable_metamorphic_testing = False  # Can be enabled if needed
        self.enable_hypothesis_testing = True     # SHT: Structured Hypothesis Testing

    # -------------------------------------------------------------------------
    # OPTIMIZED: Extract Answer with Multiple Strategies
    # -------------------------------------------------------------------------
    
    def extract_answer(self, text: Any) -> Tuple[str, Any]:
        """
        Enhanced answer extraction with multiple fallback strategies.
        """
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
        """Wrapper for _extract_last_number for backwards compatibility."""
        return _extract_last_number(s)

    # -------------------------------------------------------------------------
    # OPTIMIZED: Mathematician Agent with Structured Prompt
    # -------------------------------------------------------------------------
    
    def run_mathematician_analysis(self, problem: str) -> dict:
        """
        Enhanced Mathematician that produces a clear, actionable blueprint.
        
        Key improvements:
        - Explicit JSON schema with example
        - Requires Python-syntax equations
        - Adds solution_steps for clarity
        - Identifies distractors
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
  "distractor_check": "List any numbers/info in the problem to IGNORE (if any)"
}

CRITICAL RULES:
1. Extract ONLY relevant numbers into 'givens'
2. Use descriptive variable names (e.g., 'initial_apples', 'eaten_apples')
3. Each equation must be valid Python code
4. Equations should reference the 'givens' dict explicitly
5. solution_steps should guide step-by-step
6. If you see irrelevant information, note it in distractor_check
7. Return ONLY valid JSON, no preamble or explanation

EXAMPLE:
Problem: "Jane has 10 apples. She eats 3. How many are left?"
Output:
{
  "unknown": "number of apples remaining",
  "givens": {"initial_apples": 10, "eaten_apples": 3},
  "solution_steps": [
    "Step 1: Calculate remaining apples by subtracting eaten from initial"
  ],
  "equations": [
    "answer = givens['initial_apples'] - givens['eaten_apples']"
  ],
  "distractor_check": "None"
}
"""
        
        msgs = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": f"Problem:\n{problem}\n\nAnalyze and return the JSON blueprint."}
        ]
        
        # OPTIMIZED: Increased tokens for complex problems
        res = self.client.call_model(msgs, temperature=self.math_temp, max_tokens=1200)
        blueprint = _extract_blueprint_json(str(res))
        
        return blueprint

    # -------------------------------------------------------------------------
    # OPTIMIZED: Programmer Agent with Clear Instructions
    # -------------------------------------------------------------------------
    
    def run_programmer_solver(self, problem: str, blueprint: dict, max_attempts: int = 3) -> AgentResponse:
        """
        Enhanced Programmer that strictly follows the Architect's blueprint.
        
        Key improvements:
        - Clear instructions to implement each equation
        - Shows explicit example
        - Better error messages
        - Repair loop with specific feedback
        """
        
        givens = blueprint.get("givens", {})
        equations = blueprint.get("equations", [])
        solution_steps = blueprint.get("solution_steps", [])
        unknown = blueprint.get("unknown", "the answer")
        
        # OPTIMIZED: Format blueprint in a clear, structured way
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
        
        # OPTIMIZED: Repair loop with specific feedback
        repair_feedback = ""
        best_answer = None
        last_code = None
        
        for attempt in range(max_attempts):
            msgs = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg + repair_feedback}
            ]
            
            # OPTIMIZED: Increased tokens, lower temperature
            raw_response = self.client.call_model(
                msgs, 
                temperature=self.prog_temp, 
                max_tokens=1500
            )
            
            # Check for generation error
            if str(raw_response).startswith("ERROR_GENERATION"):
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
            
            # Optional: Metamorphic testing (non-blocking)
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
        
        # Failed after 3 attempts
        fallback_answer = best_answer if best_answer else "unknown"
        return AgentResponse(
            agent="Programmer (failed)",
            answer=fallback_answer,
            parsed=fallback_answer,
            confidence=0.2,
            reasoning_trace=last_code[:500] if last_code else "No code generated",
            quality_metrics={
                "error": "Max attempts reached",
                "last_feedback": repair_feedback
            }
        )

    # -------------------------------------------------------------------------
    # Metamorphic Testing (Optional, Currently Disabled)
    # -------------------------------------------------------------------------
    
    def _metamorphic_gate(self, code_block: str, tests: list) -> Tuple[bool, str]:
        """
        Optional metamorphic testing.
        Set self.enable_metamorphic_testing = True to enable.
        """
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
        for test in tests[:3]:  # Limit to 3 tests
            name = test.get("name", "unnamed")
            muts = test.get("mutations", [])
            assertion = test.get("assert", {})
            
            # Apply mutations
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
            
            # Run mutated code
            mutated_code = _replace_givens_dict_in_code(code_block, mutated_givens)
            ok2, out2 = PythonExecutor.execute(mutated_code)
            if not ok2:
                logs.append(f"[{name}] SKIP: Execution failed")
                continue
            
            val2 = _extract_last_number(out2)
            if val2 is None:
                logs.append(f"[{name}] SKIP: Output not numeric")
                continue
            
            # Check assertion
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

    # -------------------------------------------------------------------------
    # SHT Step 1: Confidence Gate (rule-based, 0 API calls)
    # -------------------------------------------------------------------------

    def _confidence_gate(self, primary_answer: str, baseline_answer: str,
                         programmer_response: AgentResponse,
                         blueprint: dict) -> Tuple[bool, str]:
        """
        Decide whether to trigger hypothesis testing.
        Returns (is_confident, reason).
        If is_confident=True, skip SHT. If False, trigger SHT.
        """
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
        attempts = programmer_response.quality_metrics.get("attempts", 1)
        if programmer_response.quality_metrics.get("error") == "Max attempts reached":
            return False, "max_attempts_exhausted"

        # Criterion 4: Sanity checks on the answer
        if primary_num is not None:
            # 4a: Negative answer for a word problem (usually expects positive)
            if primary_num < 0:
                return False, "negative_answer"

            # 4b: Answer is extremely large relative to givens
            givens = blueprint.get("givens", {})
            if givens:
                max_given = max(
                    (abs(v) for v in givens.values() if isinstance(v, (int, float))),
                    default=0
                )
                if max_given > 0 and abs(primary_num) > max_given * 10000:
                    return False, "answer_magnitude_suspicious"

        # All checks passed — confident
        return True, "all_checks_passed"

    # -------------------------------------------------------------------------
    # SHT Step 2: Alternative Hypothesis Generator (1 API call)
    # -------------------------------------------------------------------------

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
        Generate 2 structurally different solution blueprints.
        Each uses a different mathematical strategy archetype.
        Returns a list of blueprint dicts.
        """
        primary_strategy = primary_blueprint.get("notes", "")
        primary_eqs = primary_blueprint.get("equations", [])

        sys_msg = f"""You are an expert Mathematician who generates ALTERNATIVE solution strategies.

You have already seen one solution approach. Now produce exactly 2 DIFFERENT approaches
using DIFFERENT mathematical strategies.

AVAILABLE STRATEGY ARCHETYPES:
{chr(10).join(f'  {i+1}. {s}' for i, s in enumerate(self.STRATEGY_ARCHETYPES))}

CRITICAL RULES:
1. Each alternative MUST use a genuinely different strategy archetype
2. Do NOT repeat the same equations or approach in different words
3. Each alternative must be a complete, self-contained solution plan
4. All equations must be valid Python code referencing givens['key']
5. Return ONLY valid JSON, no preamble

OUTPUT FORMAT (strict JSON):
{{
  "alternatives": [
    {{
      "strategy_name": "one of the archetype names above",
      "unknown": "what we need to find",
      "givens": {{"var_name": numeric_value, ...}},
      "solution_steps": ["Step 1: ...", "Step 2: ..."],
      "equations": ["step1 = givens['var'] ...", "answer = ..."]
    }},
    {{
      "strategy_name": "a DIFFERENT archetype name",
      "unknown": "what we need to find",
      "givens": {{"var_name": numeric_value, ...}},
      "solution_steps": ["Step 1: ...", "Step 2: ..."],
      "equations": ["step1 = givens['var'] ...", "answer = ..."]
    }}
  ]
}}"""

        user_msg = f"""PROBLEM:
{problem}

EXISTING SOLUTION (use a DIFFERENT approach):
Strategy: {primary_strategy}
Equations: {json.dumps(primary_eqs)}
Answer obtained: {primary_answer}

Generate 2 alternative solution strategies as JSON."""

        msgs = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}
        ]

        raw = self.client.call_model(msgs, temperature=0.2, max_tokens=1800)

        # Parse the response
        alternatives = []
        try:
            text = str(raw).strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()

            # Try direct JSON parse
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
                for alt in parsed["alternatives"][:2]:
                    if isinstance(alt, dict):
                        alt.setdefault("unknown", "the answer")
                        alt.setdefault("givens", {})
                        alt.setdefault("solution_steps", [])
                        alt.setdefault("equations", [])
                        alt.setdefault("strategy_name", "unknown_strategy")
                        alternatives.append(alt)
        except Exception as e:
            logger.warning(f"SHT: Failed to parse alternative hypotheses: {e}")

        return alternatives

    # -------------------------------------------------------------------------
    # SHT Step 3: Triage Candidates (rule-based, 0 API calls)
    # -------------------------------------------------------------------------

    def _triage_candidates(self, candidates: List[HypothesisResult]) -> Tuple[Optional[str], Optional[str], str]:
        """
        Rule-based voting among candidates.
        Returns (winning_answer, winning_strategy, triage_method).
        triage_method is one of: "unanimous", "majority", "no_majority"
        """
        # Filter to successful candidates with numeric answers
        valid = [c for c in candidates if c.code_success and c.parsed_answer is not None]

        if not valid:
            return None, None, "no_valid_candidates"

        # Group by numeric answer (tolerance 1e-3)
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

        # Sort groups by size (descending), then by average confidence
        sorted_groups = sorted(
            groups.items(),
            key=lambda g: (len(g[1]), sum(c.confidence for c in g[1]) / len(g[1])),
            reverse=True
        )

        best_answer_key, best_group = sorted_groups[0]
        total_valid = len(valid)

        # Unanimous agreement
        if len(sorted_groups) == 1:
            winner = best_group[0]
            return winner.answer, winner.strategy_name, "unanimous"

        # Majority (>= 2 out of valid candidates, and strictly more than any other group)
        if len(best_group) >= 2 and (len(sorted_groups) < 2 or len(best_group) > len(sorted_groups[1][1])):
            winner = best_group[0]
            return winner.answer, winner.strategy_name, "majority"

        # No clear majority
        return None, None, "no_majority"

    # -------------------------------------------------------------------------
    # SHT Step 4: Judge Agent (0-1 API calls, only when no majority)
    # -------------------------------------------------------------------------

    def _judge_hypotheses(self, problem: str,
                          candidates: List[HypothesisResult]) -> Tuple[str, str, str]:
        """
        LLM-based Judge that evaluates reasoning quality when triage fails.
        Returns (final_answer, winning_strategy, judge_reasoning).
        """
        # Build candidate summaries for the Judge
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

        raw = self.client.call_model(msgs, temperature=0.0, max_tokens=800)
        raw_text = str(raw)

        # Extract judge's selected answer
        answer_match = re.search(r'SELECTED_ANSWER:\s*\[\[([^\]]+)\]\]', raw_text)
        strategy_match = re.search(r'SELECTED_STRATEGY:\s*\[\[([^\]]+)\]\]', raw_text)

        if answer_match:
            judge_answer = answer_match.group(1).strip()
        else:
            # Fallback: extract last number from judge response
            num = _extract_last_number(raw_text)
            judge_answer = str(num) if num is not None else "unknown"

        judge_strategy = strategy_match.group(1).strip() if strategy_match else "judge_selection"

        return judge_answer, judge_strategy, raw_text[:500]

    # -------------------------------------------------------------------------
    # SHT Step 5: Orchestrator
    # -------------------------------------------------------------------------

    def _structured_hypothesis_testing(self, problem: str, expected: str,
                                       primary_blueprint: dict,
                                       programmer_response: AgentResponse,
                                       baseline_answer: str) -> HypothesisLog:
        """
        Main SHT orchestrator. Coordinates confidence gate, hypothesis
        generation, execution, triage, and judging.
        """
        primary_answer = programmer_response.answer
        primary_num = _extract_last_number(primary_answer)

        log = HypothesisLog(
            problem=problem,
            expected=expected,
            final_answer=primary_answer,
            final_strategy="primary",
        )

        # Record primary as first candidate
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

        # Record baseline as a candidate too
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

        # --- Confidence Gate ---
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

        # --- Trigger SHT ---
        logger.info(f"SHT triggered: {gate_reason}")
        log.hypothesis_testing_triggered = True
        api_calls = 3  # baseline + mathematician + programmer already used

        # Generate 2 alternative blueprints (1 API call)
        alt_blueprints = self.generate_alternative_hypotheses(
            problem, primary_blueprint, primary_answer
        )
        api_calls += 1

        # Execute each alternative through the Programmer (1 API call each, 1 attempt)
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

        # --- Triage (rule-based, 0 API calls) ---
        triage_answer, triage_strategy, triage_method = self._triage_candidates(log.candidates)

        if triage_method in ("unanimous", "majority"):
            log.triage_result = triage_method
            log.final_answer = triage_answer
            log.final_strategy = triage_strategy
            log.api_calls_used = api_calls
            return log

        # --- Judge (1 API call, only when no majority) ---
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
        """
        Main solve method with Structured Hypothesis Testing (SHT).

        Flow:
        1. Run baseline (zero-shot)
        2. Run Architect (Mathematician)
        3. Run Engineer (Programmer)
        4. SHT: Confidence gate → alternative hypotheses → triage → judge
        5. Fallback to baseline if everything fails
        """

        # Step 1: Baseline
        baseline_prompt = f"{problem}\n\nSolve this step-by-step. End with: ANSWER: [[numeric_value]]"
        base_raw = self.client.call_model(
            [{"role": "user", "content": baseline_prompt}],
            temperature=0.1,
            max_tokens=800
        )
        base_ans, _ = self.extract_answer(base_raw)

        # Step 2: Architect
        blackboard_logic = self.run_mathematician_analysis(problem)

        # Step 3: Engineer
        programmer_response = self.run_programmer_solver(problem, blackboard_logic)

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

        # Step 5: Fallback logic (applies if SHT still returns "unknown")
        if self.enable_baseline_fallback_on_mas_failure:
            if str(mas_answer).strip().lower() == "unknown" and str(base_ans).strip().lower() != "unknown":
                mas_answer = base_ans
                used_baseline_fallback = True

        result = {
            "problem": problem,
            "expected": expected,
            "baseline": {"answer": base_ans},
            "mas": {
                "answer": mas_answer,
                "logic_trace": json.dumps(blackboard_logic, ensure_ascii=False)[:500],
                "used_baseline_fallback": used_baseline_fallback,
                "programmer_metrics": programmer_response.quality_metrics
            },
            "agents": [programmer_response],
        }

        # Attach SHT metadata
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
    def __init__(self, provider: str = "groq", use_cache: bool = False):
        self.client = UnifiedLLMClient(provider, use_cache=use_cache)
        self.manager = EnhancedProblemManager(random_seed=None)
        self.solver = QualityEnhancedMultiAgentSolver(self.client)
        self.results: List[Dict[str, Any]] = []

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
        logger.info(f"Pipeline started. Fetching {num_problems} random problems from: {datasets_list} | hardener={hardener}")
        problems = self.manager.load_random_problems(datasets_list, num_problems, hardener=hardener)

        detailed = []
        for i, p in enumerate(problems):
            logger.info(f"Processing {i+1}/{len(problems)} (ID: {p['id']}, DS: {p['dataset']})")
            res = self.solver.solve(p["puzzle"], p["answer"])
            res["baseline"]["correct"] = self.check_correctness(res["baseline"]["answer"], p["answer"])
            res["mas"]["correct"] = self.check_correctness(res["mas"]["answer"], p["answer"])
            res["id"] = p["id"]
            res["dataset"] = p["dataset"]
            detailed.append(res)

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
                **sht_data[i],
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

        # SHT rescue/damage analysis
        rescue_count = 0
        damage_count = 0
        for r in self.results:
            sht = r.get("sht", {})
            if not sht.get("triggered", False):
                continue
            # Compare: would the primary (pre-SHT) answer have been correct?
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
        print(f"Total Examples: {n}")
        print("-" * 60)
        print(f"{'Metric':<30} | {'Value':<10}")
        print("-" * 60)
        print(f"{'Baseline Accuracy':<30} | {base_acc:.2%}")
        print(f"{'MAS+SHT Accuracy':<30} | {mas_acc:.2%}")
        print(f"{'Improvement over Baseline':<30} | {(mas_acc - base_acc):+.2%}")
        print(f"{'MAS->Baseline Fallback':<30} | {fb_rate:.2%}")
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
    print("  Multi-Agent Math Solver - VERSION 7.0")
    print("  Architect-Engineer + Structured Hypothesis Testing (SHT)")
    print("  Key Features:")
    print("  ✓ Structured Mathematician prompt with JSON schema + example")
    print("  ✓ Clear Programmer instructions that follow blueprint")
    print("  ✓ Confidence gate detects uncertain answers")
    print("  ✓ Alternative hypothesis generation (2 diverse strategies)")
    print("  ✓ Rule-based triage + LLM Judge for final selection")
    print("=" * 70)
    print()

    print("Select Provider:")
    print("1) Groq")
    print("2) Google")
    choice = input("Enter selection (1 or 2): ").strip()
    prov = "google" if choice == "2" else "groq"

    pipeline = QualityAwarePipeline(provider=prov, use_cache=False)

    df_results = pipeline.run(
        datasets_list=["gsm-plus", "gsm-symbolic-p2", "gsm-hard", "svamp", "gsm8k_test"],
        num_problems=60,
        hardener="distractor",
    )
    pipeline.report()
    df_results.to_csv("final_results_sht.csv", index=False)
    print("Results saved to 'final_results_sht.csv'.")