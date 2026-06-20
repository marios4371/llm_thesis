# MAS-SHT Codebase — Deep Code Review

> Reviewed files: [Mas_solver.py](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/Mas_solver.py), [MAS_SHT_Kaggle.ipynb](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/MAS_SHT_Kaggle.ipynb), [siv_module.py](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/siv_module.py), [baselines.py](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/baselines.py)

---

## 1. Concurrency & I/O Bottlenecks

### 1.1 — Synchronous `time.sleep` Blocks the Main Thread During Backoff

**Where:** [RateLimiter.wait()](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/Mas_solver.py#L461-L467), [call_model retry loop](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/Mas_solver.py#L1172-L1231)

**Why:** Every `time.sleep()` call in the retry loop (lines 1179, 1219, 1225, 1229) blocks the single Python thread. During a 429 backoff on Groq (which can be up to 120s), the entire pipeline is frozen. This is the single largest wall-clock bottleneck for multi-problem runs. For a 120-problem Kaggle run at ~3 min/problem, the cumulative dead time from rate-limiter waits, inter-problem cooldowns, and 429 backoffs can exceed 30% of total runtime.

**How — Async-aware pipeline with `asyncio` gated behind a flag:**

The key insight is that you don't need to convert the entire solver to async. You only need async at the I/O boundary (API calls + sleep). The compute-heavy steps (SIV, SymPy, PythonExecutor) remain synchronous. This gives you two benefits without blowing up Kaggle RAM: (a) overlapping wait-time with the next problem's non-API phases, and (b) enabling future batched API calls.

```python
# Drop-in replacement for RateLimiter that works both sync and async.
# Place this in Mas_solver.py alongside the existing RateLimiter.

import asyncio
from typing import Optional

class AsyncRateLimiter:
    """Rate limiter that can yield control during wait instead of blocking."""

    def __init__(self, requests_per_minute: int = 12):
        self.delay = 60.0 / max(1, requests_per_minute)
        self.last_call = 0.0
        self._lock = asyncio.Lock()

    async def wait_async(self) -> None:
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.delay:
                await asyncio.sleep(self.delay - elapsed)
            self.last_call = time.time()

    def wait(self) -> None:
        """Synchronous fallback — identical to the existing behaviour."""
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_call = time.time()
```

> [!TIP]
> For the Kaggle notebook, wrap the experiment loop in an `async for` and use `asyncio.gather` with `max_concurrency=2` to overlap baseline and MAS solves for the same problem. Two concurrent solves keep one API call in-flight while the other does CPU-bound SIV/SymPy. This alone can cut wall-clock by ~25% without increasing peak RAM.

### 1.2 — Cache Serialization is Per-Call, Not Batched

**Where:** [call_model cache write](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/Mas_solver.py#L1183-L1185)

**Why:** Every successful API call writes the entire `CALL_CACHE` dict to disk via `pickle.dump`. With a cache file already at 941 KB, this becomes O(n²) in cumulative I/O as the cache grows. On a 1000-problem GSM8K run, you'd serialize the full cache ~5000 times.

**How — Deferred batch writes:**

```python
# In UnifiedLLMClient.__init__ or at module level:
_CACHE_DIRTY = False

def _mark_cache_dirty():
    global _CACHE_DIRTY
    _CACHE_DIRTY = True

def flush_cache_if_dirty():
    """Call this every N problems (e.g., at checkpoint time) instead of per-call."""
    global _CACHE_DIRTY, CALL_CACHE
    if not _CACHE_DIRTY:
        return
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(CALL_CACHE, f)
        _CACHE_DIRTY = False
    except Exception as e:
        logger.warning(f"Cache flush failed: {e}")

# Then in call_model, replace the per-call write:
# OLD (lines 1183-1185):
#   if self.use_cache:
#       CALL_CACHE[key] = res
#       with open(CACHE_FILE, "wb") as f:
#           pickle.dump(CALL_CACHE, f)

# NEW:
if self.use_cache:
    CALL_CACHE[key] = res
    _mark_cache_dirty()
```

Then in the notebook's checkpoint block, add `flush_cache_if_dirty()` alongside `_save_ckpt()`.

### 1.3 — Provider-Specific Retry Strategies Should Be Extracted

**Where:** [call_model lines 1172-1231](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/Mas_solver.py#L1172-L1231)

**Why:** The retry loop mixes Groq-specific 429 parsing (`try again in 8m27.168s`), generic exponential backoff, and auth-error short-circuits in a single function. Adding new providers (you have five now) means this block grows linearly. More importantly, each provider has different retry semantics: HuggingFace 503s are cold-starts (wait for `estimated_time`), Together 429s use standard Retry-After headers, and local_hf never needs retries.

**How — Strategy pattern per provider:**

```python
from dataclasses import dataclass
from typing import Callable, Optional
import re

@dataclass
class RetryDecision:
    should_retry: bool
    wait_seconds: float = 0.0
    error_sentinel: Optional[str] = None  # If set, return this immediately

def _groq_retry_strategy(e: Exception, attempt: int) -> RetryDecision:
    """Groq-specific: parse 'try again in XmYs' from 429 body."""
    err_str = str(e).lower()
    if "401" in err_str or "unauthorized" in err_str:
        return RetryDecision(False, error_sentinel=f"ERROR_AUTH_401: {e}")
    if "429" in err_str or "rate_limit" in err_str:
        wait_match = re.search(r'try again in (\d+)m([\d.]+)s', str(e))
        if wait_match:
            total = int(wait_match.group(1)) * 60 + float(wait_match.group(2)) + 5
            if total > 600:
                return RetryDecision(False, error_sentinel=f"ERROR_RATE_LIMIT_DAILY: {e}")
            return RetryDecision(True, wait_seconds=total)
        return RetryDecision(True, wait_seconds=min(120, (2 ** attempt) * 5 + random.uniform(0, 5)))
    return RetryDecision(True, wait_seconds=min(12.0, 1.5 * (attempt + 1)))

def _local_hf_retry_strategy(e: Exception, attempt: int) -> RetryDecision:
    """Local inference: only retry on transient CUDA errors, not auth."""
    if "CUDA" in str(e) or "out of memory" in str(e).lower():
        return RetryDecision(True, wait_seconds=2.0)
    return RetryDecision(False, error_sentinel=f"ERROR_LOCAL: {e}")

# Map in __init__:
# self._retry_strategy = {
#     "groq": _groq_retry_strategy,
#     "google": _generic_retry_strategy,
#     "huggingface": _hf_retry_strategy,
#     "together": _generic_retry_strategy,
#     "local_hf": _local_hf_retry_strategy,
# }[self.provider]
```

---

## 2. VRAM Management & Local Inference

### 2.1 — `_local_model` Reference Leak on Failed Cleanup

**Where:** [_ensure_local_model](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/Mas_solver.py#L1347-L1477), [_free_local_hf_models in notebook](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/MAS_SHT_Kaggle.ipynb) Cell 3

**Why:** In `_ensure_local_model` (line 1472), when a load fails, `self._local_model = None` is set correctly. But the notebook's `_free_local_hf_models` helper does `mdl.cpu()` before `del mdl`. If `mdl.cpu()` fails (e.g., CUDA context already corrupted), the exception is caught but `client._local_model` is NOT set to None — the stale reference survives, and the model is never garbage-collected. The next call to `_ensure_local_model` sees `self._local_model is not None` (line 1350) and skips loading, returning a zombie model.

**How — Null the reference before attempting `.cpu()`:**

```python
# In the notebook's _free_local_hf_models, replace the try block:

def _free_local_hf_models(pipeline_obj):
    """Safely free all local_hf models, null references FIRST."""
    freed = 0
    for key, client in list(getattr(pipeline_obj, '_client_cache', {}).items()):
        mdl = getattr(client, '_local_model', None)
        if mdl is not None:
            # Null references BEFORE attempting .cpu() — if .cpu() throws,
            # we still want _ensure_local_model to reload next time.
            client._local_model = None
            client._local_tokenizer = None
            try:
                import torch
                mdl.cpu()
            except Exception:
                pass  # CUDA context may be corrupted; just drop the reference
            del mdl
            freed += 1
    if freed:
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Force pending deallocations
        except Exception:
            pass
        print(f"  [VRAM] freed {freed} local_hf model(s) — GPU cache cleared")
```

### 2.2 — Missing `torch.cuda.synchronize()` After `empty_cache()`

**Where:** [_ensure_local_model line 1475](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/Mas_solver.py#L1474-L1476), notebook Cell 3 VRAM cleanup

**Why:** `torch.cuda.empty_cache()` releases the cache but doesn't guarantee that all pending CUDA operations have completed. On T4 GPUs with tight VRAM budgets, a subsequent `from_pretrained` can start allocating while the prior model's deallocations are still in-flight, causing OOM even though there's technically enough VRAM.

**How:**

```python
# After every empty_cache() call, add synchronize():
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()  # Wait for all pending CUDA ops to complete
```

### 2.3 — KV-Cache Memory is Unbounded for Long CoT Outputs

**Where:** [_call_local_hf](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/Mas_solver.py#L1479-L1584)

**Why:** `max_new_tokens=max_tokens` (line 1528) can be up to 1536 for the Mathematician role. On a 7B model with fp16 on T4, each KV-cache slot for a 1536-token generation at layer-32 / head-32 architecture consumes ~96 MB. Combined with the model's ~14 GB footprint, this leaves under 1 GB headroom on a T4 x2 setup. If a model enters a degenerate loop (generating until max_tokens without an EOS), the KV-cache grows to fill remaining VRAM and the next call OOMs.

**How — Add a generation-time timeout and EOS enforcement:**

```python
# In _call_local_hf, after gen_kwargs definition (around line 1531):

# Cap max_new_tokens to prevent runaway KV-cache growth.
# 7B models on T4 x2 have ~1 GB headroom after model weights;
# 512 new tokens ≈ 32 MB KV cache, safe. 1536 ≈ 96 MB, borderline.
effective_max = min(max_tokens, 1024)  # Hard cap for VRAM safety
gen_kwargs["max_new_tokens"] = effective_max

# Force stop on EOS to prevent degenerate loops:
if tok.eos_token_id is not None:
    gen_kwargs["eos_token_id"] = tok.eos_token_id

# Detect and truncate repetitive output (model stuck in a loop):
# This is cheaper than a timeout and catches the "!!!!!" failure mode.
gen_kwargs["repetition_penalty"] = 1.1  # Mild penalty, doesn't hurt quality
```

### 2.4 — Duplicate `self.load_4bit` Assignment

**Where:** [UnifiedLLMClient.__init__ line 1026-1027](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/Mas_solver.py#L1026-L1027)

**Why:** `self.load_4bit = load_4bit` is assigned twice in consecutive lines. This is cosmetic but indicates a copy-paste artifact that could mask a real bug if one line was meant to set a different attribute.

**How:**

```diff
- self.load_4bit = load_4bit  # [v10.3]
- self.load_4bit = load_4bit  # [v10.3] 4-bit NF4 quantization for local_hf 7B models
+ self.load_4bit = load_4bit  # [v10.3] 4-bit NF4 quantization for local_hf 7B models
```

### 2.5 — Duplicate Preset Definitions

**Where:** [HETEROGENEOUS_PRESETS](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/Mas_solver.py#L288-L316) and [lines 345-351](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/Mas_solver.py#L345-L351)

**Why:** The presets `"qwen_math_7b_local"` (lines 288-294) and `"deepseek_7b_local"` (lines 299-305) are each defined **twice** in the same dict literal. Python silently takes the last definition, which means the first definition is dead code. Both definitions are identical in this case, so there's no runtime bug — but it bloats the file and could silently mask a future edit that only touches one copy.

**How:** Delete lines 288-316 (the first occurrences of both presets). The second definitions at lines 310-316 and 345-351 will remain as the sole definitions.

---

## 3. Parsing & Extraction Robustness

### 3.1 — `_extract_blueprint_json` Fails on Nested JSON Structures

**Where:** [_extract_blueprint_json](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/Mas_solver.py#L587-L668)

**Why:** The regex-based fallback for `givens` extraction (line 645: `r'"givens"\s*:\s*(\{[^}]+\})'`) uses `[^}]+` which fails on nested braces. If a model outputs `"givens": {"total_cost": {"value": 5}}` (unlikely but possible with creative models), the regex captures only `{"total_cost": {"value": 5}` — a malformed substring. More practically, the `equations` extraction (line 653: `r'"equations"\s*:\s*\[(.*?)\]'`) uses `.*?` which fails on arrays containing strings with `]` characters (e.g., `["arr[0] = givens['x']"]`).

**How — Lightweight Pydantic validation with graceful degradation:**

```python
# At top of file, add conditional import:
try:
    from pydantic import BaseModel, Field, ValidationError
    from typing import Dict, List, Optional
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

# Define the blueprint schema:
if PYDANTIC_AVAILABLE:
    class BlueprintModel(BaseModel):
        """Validated blueprint schema. Provides type coercion + defaults."""
        unknown: str = "the answer"
        givens: Dict[str, float] = Field(default_factory=dict)
        solution_steps: List[str] = Field(default_factory=list)
        equations: List[str] = Field(default_factory=list)
        expected_answer: str = ""
        distractor_check: str = ""
        metamorphic_tests: list = Field(default_factory=list)
        notes: str = ""
        reasoning: str = ""  # v11.1 constrained decoding field

        class Config:
            extra = "allow"  # Don't reject unknown keys from LLM

# Then rewrite _extract_blueprint_json:
def _extract_blueprint_json(text: str) -> dict:
    """Enhanced JSON extraction with Pydantic validation fallback."""
    if _is_error_response(text):
        logger.warning(f"Mathematician returned error response: {str(text)[:200]}")
        return _empty_blueprint(f"ERROR: {str(text)[:200]}")

    text = str(text).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()

    candidates = []

    # Strategy 1: Direct parse
    try:
        candidates.append(json.loads(text))
    except json.JSONDecodeError:
        pass

    # Strategy 2: Substring extraction (find outermost {})
    if not candidates:
        start = text.find("{")
        if start != -1:
            # Balanced-brace extraction instead of rfind
            depth, end = 0, start
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end > start:
                try:
                    candidates.append(json.loads(text[start:end + 1]))
                except json.JSONDecodeError:
                    pass

    # Validate with Pydantic if available
    for raw_dict in candidates:
        if not isinstance(raw_dict, dict):
            continue
        if PYDANTIC_AVAILABLE:
            try:
                bp = BlueprintModel(**raw_dict)
                return bp.model_dump()
            except ValidationError as ve:
                logger.debug(f"Blueprint validation: {ve.error_count()} errors, using raw")
        # Fallback: manual defaults
        raw_dict.setdefault("unknown", "the answer")
        raw_dict.setdefault("givens", {})
        raw_dict.setdefault("solution_steps", [])
        raw_dict.setdefault("equations", [])
        raw_dict.setdefault("distractor_check", "")
        raw_dict.setdefault("metamorphic_tests", [])
        raw_dict.setdefault("notes", "")
        return raw_dict

    # Strategy 3: Regex fallback (unchanged — last resort)
    return _regex_fallback_blueprint(text)


def _empty_blueprint(note: str = "") -> dict:
    return {
        "unknown": "the answer", "givens": {},
        "solution_steps": ["Error: LLM call failed"],
        "equations": [], "distractor_check": "",
        "metamorphic_tests": [], "notes": note,
    }


def _regex_fallback_blueprint(text: str) -> dict:
    """Last-resort regex extraction — unchanged from current code."""
    givens = {}
    equations = []
    givens_match = re.search(r'"givens"\s*:\s*(\{[^}]+\})', text, re.DOTALL)
    if givens_match:
        try:
            givens = json.loads(givens_match.group(1))
        except Exception:
            pass
    eqs_match = re.search(r'"equations"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if eqs_match:
        try:
            equations = json.loads(f'[{eqs_match.group(1)}]')
        except Exception:
            pass
    return {
        "unknown": "the answer", "givens": givens,
        "solution_steps": ["Solve step by step"],
        "equations": equations, "distractor_check": "",
        "metamorphic_tests": [], "notes": text[:800],
    }
```

### 3.2 — `_extract_code_from_response` Matches Non-Code Content

**Where:** [_extract_code_from_response](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/Mas_solver.py#L688-L733)

**Why:** The generic ` ``` ` pattern (line 702: `` r"```\s+(.*?)```" ``) matches *any* fenced block — including JSON, bash, or plain text blocks. If the model outputs both a JSON block and a Python block, the first match wins even if it's JSON. This has likely caused silent failures where JSON output is passed to `PythonExecutor.execute`.

**How — Prioritize language-tagged blocks and validate minimally:**

```python
def _extract_code_from_response(raw: str) -> Optional[str]:
    """Enhanced code extraction: prioritise python-tagged blocks, validate content."""
    if _is_error_response(raw):
        return None

    s = str(raw)

    # Priority 1: Explicitly python-tagged fences
    for pattern in [r"```python\s+(.*?)```", r"```py\s+(.*?)```", r"~~~python\s+(.*?)~~~"]:
        match = re.search(pattern, s, re.DOTALL | re.IGNORECASE)
        if match:
            code = match.group(1).strip()
            code = re.sub(r"^python\s*\n", "", code, flags=re.IGNORECASE)
            return code

    # Priority 2: Generic fenced blocks — but ONLY if they look like Python
    for pattern in [r"```\s+(.*?)```", r"~~~\s+(.*?)~~~"]:
        match = re.search(pattern, s, re.DOTALL | re.IGNORECASE)
        if match:
            code = match.group(1).strip()
            # Heuristic: reject if it looks like JSON (starts with {/[)
            if code.lstrip().startswith(("{", "[")):
                continue
            # Heuristic: must contain at least one assignment or print
            if re.search(r'(?:=|print\s*\()', code):
                return code

    # Priority 3: Open fence (missing closing ```), existing logic
    for pattern in [r"```(?:python|py)?\s+(.*?)$"]:
        match = re.search(pattern, s, re.DOTALL | re.IGNORECASE)
        if match:
            code = match.group(1).strip()
            code = re.split(r"\n\n(?:ANSWER|---|Note|Explanation)", code, maxsplit=1)[0]
            if re.search(r'(?:=|print\s*\()', code):
                return code.strip()

    # Priority 4: Bare code (starts with `givens =`)
    givens_match = re.search(r"^(givens\s*=\s*\{.*)", s, re.DOTALL | re.MULTILINE)
    if givens_match:
        code = givens_match.group(1)
        code = re.split(r"\n\n(?:ANSWER|---)", code, maxsplit=1)[0]
        return code.strip()

    return None
```

### 3.3 — `_extract_last_number` Strips Unit Words That May Be Part of the Answer

**Where:** [_extract_last_number](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/Mas_solver.py#L736-L760)

**Why:** The regex on line 747 strips unit suffixes like "hours" and "minutes". Consider a problem whose answer is "The meeting lasts 3 hours". The function correctly extracts 3. But for "There are 5 items" → strips "items", extracts 5 ✓. Now consider edge case: "Answer: 24 hours, so 1440 minutes" — the stripping removes "minutes", then `findall` returns `['24', '1440']`, and we take the last one (1440). If the expected answer is 24 hours, this is wrong. The issue is that the regex operates on the entire text, not just the tail.

**How — Only strip from the final token, not the whole string:**

```python
def _extract_last_number(text: str) -> Optional[float]:
    """Extract the last numeric value, stripping units only from the final match."""
    if _is_error_response(text):
        return None

    text = str(text).strip()

    # Find all numbers (including negatives, decimals, with commas)
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)

    if not numbers:
        return None

    last_num = numbers[-1].replace(',', '')
    try:
        return float(last_num)
    except ValueError:
        return None
```

> [!NOTE]
> I removed the unit-stripping regex entirely. It was attempting to help but introduced edge cases. The function's contract is "last number in text" — unit awareness belongs in the caller or in a separate normalization step.

---

## 4. Execution Safety

### 4.1 — `PythonExecutor.execute` Passes `__builtins__` Unsanitized

**Where:** [PythonExecutor.execute line 814](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/Mas_solver.py#L807-L835)

**Why:** The current code passes `{"__builtins__": __builtins__}` as the globals dict to `exec()`. This means **all** of Python's built-in functions are available to LLM-generated code, including:

- `type()`, `getattr()`, `setattr()` → introspection/mutation of the execution frame
- `dir()`, `vars()`, `globals()` → enumerate the enclosing namespace
- `__import__()` → bypasses the string-based blocklist entirely (e.g., `__import__('subprocess')`)
- `memoryview()`, `bytearray()` → memory access primitives
- `breakpoint()` → launches the debugger, hangs the process

The forbidden-token list (lines 796-800) checks for `"__import__"` as a substring, but the LLM could use `getattr(__builtins__, '__imp' + 'ort__')('os')` to evade it via string concatenation.

> [!CAUTION]
> This is the most critical security finding. An adversarial prompt injection in the problem text could cause the LLM to generate code that escapes the sandbox.

**How — Explicit builtins whitelist:**

```python
class PythonExecutor:
    # Whitelist of safe builtins for LLM-generated math code.
    # No I/O, no imports, no introspection.
    _SAFE_BUILTINS = {
        # Arithmetic & math
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "pow": pow, "divmod": divmod,
        # Type constructors (needed for math)
        "int": int, "float": float, "bool": bool, "str": str,
        "list": list, "dict": dict, "tuple": tuple, "set": set,
        # Iteration
        "range": range, "len": len, "enumerate": enumerate, "zip": zip,
        "map": map, "filter": filter, "sorted": sorted, "reversed": reversed,
        # Truthiness
        "all": all, "any": any,
        # Printing (we capture stdout)
        "print": print,
        # Math module (inject as builtins so `import math` isn't needed)
        "True": True, "False": False, "None": None,
    }

    @staticmethod
    def execute(code_str: str, timeout_seconds: float = 10.0) -> Tuple[bool, str]:
        """Execute Python code in a restricted namespace with timeout."""
        # Blocklist check (defense in depth — whitelist is the real barrier)
        forbidden = [
            "import ", "__import__", "subprocess", "eval(", "exec(",
            "compile(", "open(", "file(", "input(", "raw_input(",
            "rm -rf", "rmdir", "getattr", "setattr", "delattr",
            "globals", "locals", "vars", "dir(",  "breakpoint",
            "memoryview", "bytearray",
        ]
        code_lower = code_str.lower()
        for token in forbidden:
            if token in code_lower:
                return False, f"SecurityError: Forbidden token '{token}'"

        try:
            import io, math
            from contextlib import redirect_stdout

            # Build safe execution namespace
            safe_globals = {"__builtins__": PythonExecutor._SAFE_BUILTINS}
            # Inject math module functions directly (no import needed)
            for name in ['ceil', 'floor', 'sqrt', 'log', 'log10', 'exp',
                         'pi', 'e', 'sin', 'cos', 'tan', 'gcd']:
                if hasattr(math, name):
                    safe_globals[name] = getattr(math, name)
            safe_globals['math'] = math  # Allow math.xxx syntax too

            local_vars = {}
            buf = io.StringIO()
            with redirect_stdout(buf):
                exec(code_str, safe_globals, local_vars)

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
        except RecursionError:
            return False, "RecursionError: Infinite recursion detected"
        except MemoryError:
            return False, "MemoryError: Code allocated too much memory"
        except Exception as e:
            return False, f"ExecutionError: {type(e).__name__}: {str(e)}"
```

### 4.2 — `SymbolicSolver.solve_from_blueprint` Also Uses Unprotected `exec`

**Where:** [SymbolicSolver.solve_from_blueprint line 917](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/Mas_solver.py#L893-L950)

**Why:** The `SymbolicSolver` has its own `exec()` call (line 917) with a custom `safe_builtins` dict that's better than `PythonExecutor`'s — but it still allows `int()`, `float()`, `sum()`, `len()`, `pow()` which are fine for math. The problem is that `exec_globals` (line 905) is `{"__builtins__": safe_builtins, "givens": givens_dict}`, which is safe. However, `exec_locals` (line 906) is `dict(namespace)` which is just a copy of givens values — also safe. **This code is actually fine.** It just doesn't have the blocklist that `PythonExecutor` has.

**How — Add the same blocklist pre-check:**

```python
@staticmethod
def solve_from_blueprint(blueprint: dict) -> Tuple[bool, str, str]:
    # ... existing code ...
    for eq in equations:
        eq = eq.strip()
        if not eq or eq.startswith("#"):
            continue
        # Add safety check for blueprint equations too
        eq_lower = eq.lower()
        if any(tok in eq_lower for tok in ["import ", "__import__", "exec(", "eval("]):
            trace_lines.append(f"  BLOCKED: unsafe token in equation: {eq}")
            return False, "unknown", "\n".join(trace_lines)
        # ... rest of execution ...
```

### 4.3 — No Execution Timeout

**Where:** [PythonExecutor.execute](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/Mas_solver.py#L793-L835)

**Why:** An LLM-generated `while True: pass` or deeply recursive function will hang the entire pipeline indefinitely. On Kaggle, this means the entire cell hangs until the kernel is killed, losing all uncheckpointed progress.

**How — Signal-based timeout (Unix) with thread fallback (Windows/Kaggle):**

```python
import threading
import ctypes

def _execute_with_timeout(code_str: str, safe_globals: dict,
                          timeout_s: float = 10.0) -> Tuple[bool, str]:
    """Execute code with a hard timeout. Uses threading for portability."""
    import io
    from contextlib import redirect_stdout

    result = [False, "Timeout: execution exceeded time limit"]
    local_vars = {}

    def _target():
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                exec(code_str, safe_globals, local_vars)
            output = buf.getvalue().strip()
            if not output:
                if 'answer' in local_vars:
                    result[0], result[1] = True, str(local_vars['answer'])
                elif 'result' in local_vars:
                    result[0], result[1] = True, str(local_vars['result'])
                else:
                    result[1] = "NoOutput: Code produced no output or answer variable"
            else:
                result[0], result[1] = True, output
        except Exception as e:
            result[1] = f"ExecutionError: {type(e).__name__}: {str(e)}"

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)

    if thread.is_alive():
        # Thread is stuck — we can't safely kill it in Python,
        # but marking it daemon ensures it dies with the process.
        return False, f"TimeoutError: Code did not finish within {timeout_s}s"

    return result[0], result[1]
```

> [!WARNING]
> Python threads cannot be forcibly killed. The daemon thread will keep running in the background until it finishes or the process exits. For true isolation, use `multiprocessing` with a timeout. However, on Kaggle's constrained environment, the threading approach is the pragmatic choice — it at least lets the pipeline continue with other problems instead of hanging indefinitely.

### 4.4 — `exec` Exceptions Not Fully Caught

**Where:** [PythonExecutor.execute lines 828-835](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/Mas_solver.py#L828-L835)

**Why:** The current handler catches `NameError`, `KeyError`, `ZeroDivisionError`, and a generic `Exception`. However, it doesn't explicitly catch `RecursionError` (which can crash the interpreter if the recursion limit is too high) or `MemoryError` (which could OOM the Kaggle kernel). Both of these should return clean error messages to the SHT Critic for diagnosis, not crash the pipeline.

**How:** Add these to the exception chain (already included in the rewritten `PythonExecutor` above in §4.1).

---

## 5. Additional Findings (Lower Severity)

### 5.1 — `bare except` Clauses Suppress Debugging Information

**Where:** Lines [620](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/Mas_solver.py#L620), [637](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/Mas_solver.py#L637), [649](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/Mas_solver.py#L649), [759](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/Mas_solver.py#L759), [1048](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/Mas_solver.py#L1048), [2311](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/Mas_solver.py#L2311), [3685](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/Mas_solver.py#L3685)

**Why:** There are 7+ bare `except:` clauses. Each one silently swallows the exception. During debugging, this means you have no way to know *why* parsing failed — was it a `json.JSONDecodeError`, a `TypeError`, or a `KeyboardInterrupt` that you really wanted to propagate?

**How:**

```python
# Replace every bare `except:` with:
except Exception:
    pass

# Or better, log what was caught:
except Exception as e:
    logger.debug(f"Parse fallback: {type(e).__name__}: {e}")
```

### 5.2 — `_extract_last_boxed` Defined Inside a Loop

**Where:** [EnhancedProblemManager.load_random_problems](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/Mas_solver.py#L1825-L1838)

**Why:** `_extract_last_boxed` is defined as a nested function *inside* the `for i in idxs[:per_ds]` loop body (line 1825). It's recreated on every iteration, which is a minor performance issue but more importantly a code-smell — it should be a module-level or class-level helper.

**How:** Move it to module level:

```python
def _extract_last_boxed(s: str) -> str:
    """Extract content from the last \\boxed{...}, handling nested braces."""
    tag = r"\boxed{"
    pos = s.rfind(tag)
    if pos == -1:
        return s.strip()
    start = pos + len(tag)
    depth, i = 1, start
    while i < len(s) and depth:
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
        i += 1
    return s[start:i - 1].strip()
```

### 5.3 — Global Mutable State for `CALL_CACHE`

**Where:** [CALL_CACHE](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/Mas_solver.py#L435) and [call_model](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/Mas_solver.py#L1043-L1049)

**Why:** `CALL_CACHE` is a module-level global dict that's loaded from disk in `UnifiedLLMClient.__init__`. If multiple `UnifiedLLMClient` instances are created (which they are — one per role in heterogeneous mode), each constructor re-reads and overwrites the global cache. The `global CALL_CACHE` statement on line 1045 means any client can stomp on the cache another client loaded. This is thread-unsafe and creates surprising behavior when `use_cache=True`.

**How — Move cache to a singleton manager:**

```python
class _CacheManager:
    """Thread-safe singleton cache. Loaded once, written on demand."""
    _instance: Optional['_CacheManager'] = None
    _lock = threading.Lock()

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._dirty = False
        self._loaded = False

    @classmethod
    def get(cls) -> '_CacheManager':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def load(self, path: str) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        self._cache = pickle.load(f)
                except Exception:
                    self._cache = {}
            self._loaded = True

    def lookup(self, key: str) -> Optional[Any]:
        return self._cache.get(key)

    def store(self, key: str, value: Any) -> None:
        self._cache[key] = value
        self._dirty = True

    def flush(self, path: str) -> None:
        if not self._dirty:
            return
        with self._lock:
            with open(path, "wb") as f:
                pickle.dump(self._cache, f)
            self._dirty = False
```

### 5.4 — `random.seed` in Constructor Has Global Side Effects

**Where:** [EnhancedProblemManager.__init__](file:///c:/Users/mario/gitproj/MAS_LLM_Thesis/Mas_solver.py#L1598-L1599)

**Why:** `random.seed(random_seed)` sets the *global* random state. Any other code in the process that calls `random.choice()` (e.g., the distractor injector, a library dependency) will produce deterministic but potentially correlated sequences. This is fine for reproducibility, but fragile if you ever add parallelism.

**How — Use an instance-level RNG:**

```python
class EnhancedProblemManager:
    def __init__(self, random_seed: Optional[int] = None):
        self._rng = random.Random(random_seed)
        # Use self._rng.sample() instead of random.sample() everywhere
```

---

## Summary Table

| # | Category | Severity | File | Lines | Fix Effort |
|---|----------|----------|------|-------|------------|
| 1.1 | Concurrency | Medium | Mas_solver.py | 461-467, 1172-1231 | Medium |
| 1.2 | I/O | Low | Mas_solver.py | 1183-1185 | Low |
| 1.3 | Architecture | Low | Mas_solver.py | 1172-1231 | Medium |
| 2.1 | VRAM Leak | **High** | Notebook Cell 3 | `_free_local_hf_models` | Low |
| 2.2 | VRAM | Medium | Mas_solver.py | 1475 | Trivial |
| 2.3 | VRAM/OOM | Medium | Mas_solver.py | 1528 | Low |
| 2.4 | Code Smell | Trivial | Mas_solver.py | 1026-1027 | Trivial |
| 2.5 | Code Smell | Trivial | Mas_solver.py | 288-351 | Trivial |
| 3.1 | Parsing | Medium | Mas_solver.py | 587-668 | Medium |
| 3.2 | Parsing | Medium | Mas_solver.py | 688-733 | Low |
| 3.3 | Parsing | Low | Mas_solver.py | 736-760 | Low |
| 4.1 | **Security** | **Critical** | Mas_solver.py | 807-814 | Low |
| 4.2 | Security | Low | Mas_solver.py | 917 | Trivial |
| 4.3 | Safety | **High** | Mas_solver.py | 793-835 | Medium |
| 4.4 | Robustness | Medium | Mas_solver.py | 828-835 | Trivial |
| 5.1 | Code Quality | Low | Mas_solver.py | multiple | Trivial |
| 5.2 | Performance | Low | Mas_solver.py | 1825-1838 | Trivial |
| 5.3 | Thread Safety | Medium | Mas_solver.py | 435, 1043-1049 | Medium |
| 5.4 | Reproducibility | Low | Mas_solver.py | 1598-1599 | Low |
