"""
[v40.0] A small Ollama client for the cross-model runs (pretest_v40.py --backend ollama).

Standard library only (urllib), so it runs on the laptop that runs Ollama with
no extra install. Two operations, the same two the HF path has:
  sample(content, k, seeds)         k independent sampled completions of one
                                    user turn (v38's knobs: T 0.8, top-p 0.95,
                                    top-k 50), each with its own seed, sent
                                    concurrently; the server decides how many
                                    it decodes at once (OLLAMA_NUM_PARALLEL)
  continue_from(content, prefix)    one continuation of an assistant turn that
                                    already begins with `prefix`: the chat
                                    request ends with that assistant message,
                                    which Ollama's templates leave open (prefill)
A model whose template does not leave the last assistant message open would
start a fresh answer instead; smoke_prefill() detects that before the run, and
the runner then records the model as unable to resume.
Thinking models are asked not to think (think: false), so that every model is
measured on the same visible chain of thought.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence, Tuple

HOST = 'http://localhost:11434'
TIMEOUT = 1800            # seconds per request (queued requests wait for the model)
RETRIES = 4               # a dropped connection (Ollama restarting) is retried with back-off


def _post(url: str, payload: Optional[Dict], timeout: float) -> Dict:
    """POST `payload` as JSON (GET when it is None) and return the JSON answer."""
    data = None if payload is None else json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    provider = 'ollama'

    def __init__(self, model: str, host: str = HOST, num_ctx: int = 4096, top_p: float = 0.95, top_k: int = 50,
                 concurrency: int = 5, transport: Optional[Callable[[str, Optional[Dict], float], Dict]] = None):
        self.model, self.host = model, host.rstrip('/')
        self.num_ctx, self.top_p, self.top_k, self.concurrency = num_ctx, top_p, top_k, concurrency
        self._post = transport or _post
        self._show: Optional[Dict] = None
        self.stats = {'calls': 0, 'eval_tokens': 0, 'eval_seconds': 0.0, 'truncated': 0}

    # ------------------------------------------------------------------ plumbing
    def _call(self, path: str, payload: Optional[Dict]) -> Dict:
        err = None
        for attempt in range(RETRIES + 1):
            try:
                return self._post(self.host + path, payload, TIMEOUT)
            except urllib.error.HTTPError as exc:            # the server answered: a real error, do not retry
                body = exc.read().decode('utf-8', 'replace')[:300] if hasattr(exc, 'read') else ''
                raise OllamaError(f"{path} -> HTTP {exc.code}: {body}") from exc
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
                err = exc
                if attempt < RETRIES:
                    time.sleep(min(60, 5 * 2 ** attempt))
        raise OllamaError(f"{path}: Ollama unreachable at {self.host} after {RETRIES + 1} tries ({err})")

    def show(self) -> Dict:
        if self._show is None:
            self._show = self._call('/api/show', {'model': self.model})
        return self._show

    def version(self) -> str:
        try:
            return self._call('/api/version', None).get('version', '?')
        except OllamaError:
            return '?'

    def thinking(self) -> bool:
        return 'thinking' in (self.show().get('capabilities') or [])

    def details(self) -> Dict:
        d = self.show().get('details') or {}
        return {k: d.get(k) for k in ('family', 'parameter_size', 'quantization_level')}

    def _options(self, temperature: float, max_tokens: int, seed: Optional[int]) -> Dict:
        o = {'temperature': float(temperature), 'top_p': self.top_p, 'top_k': self.top_k,
             'num_predict': int(max_tokens), 'num_ctx': self.num_ctx, 'repeat_penalty': 1.0}
        if seed is not None:
            o['seed'] = int(seed)
        return o

    def chat(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int,
             seed: Optional[int] = None) -> Tuple[str, Dict]:
        payload = {'model': self.model, 'messages': messages, 'stream': False, 'keep_alive': '30m',
                   'options': self._options(temperature, max_tokens, seed)}
        if self.thinking():
            payload['think'] = False
        r = self._call('/api/chat', payload)
        text = ((r.get('message') or {}).get('content') or '')
        info = {'tokens': r.get('eval_count'), 'seconds': (r.get('eval_duration') or 0) / 1e9,
                'done_reason': r.get('done_reason')}
        self.stats['calls'] += 1
        self.stats['eval_tokens'] += info['tokens'] or 0
        self.stats['eval_seconds'] += info['seconds']
        self.stats['truncated'] += info['done_reason'] == 'length'
        return text, info

    # ------------------------------------------------------------------ the two operations
    def sample(self, content: str, k: int, temperature: float, max_tokens: int,
               seeds: Sequence[Optional[int]] = ()) -> List[Tuple[str, Dict]]:
        seeds = list(seeds) + [None] * (k - len(seeds))
        msgs = [{'role': 'user', 'content': content}]
        with ThreadPoolExecutor(max_workers=max(1, min(self.concurrency, k))) as pool:
            return list(pool.map(lambda s: self.chat(msgs, temperature, max_tokens, s), seeds[:k]))

    def continue_from(self, content: str, prefix: str, temperature: float, max_tokens: int,
                      seed: Optional[int] = None) -> Tuple[str, Dict]:
        """The FULL text, prefix + continuation."""
        msgs = [{'role': 'user', 'content': content}, {'role': 'assistant', 'content': prefix}]
        text, info = self.chat(msgs, temperature, max_tokens, seed)
        return prefix + text, info

    def unload(self) -> None:
        try:
            self._call('/api/generate', {'model': self.model, 'keep_alive': 0})
        except OllamaError:
            pass


def smoke_prefill(client: OllamaClient, prompt_template: str) -> Tuple[bool, str]:
    """Does the model CONTINUE an assistant prefix (True) or start a fresh answer?
    The prefix ends right before the answer, so a continuation names 150 at once."""
    prefix = "We add the parts: 61 + 8 + 13 + 10 + 58 = 150.\nSo the final answer is"
    full, _ = client.continue_from(prompt_template.format(problem='What is 61 + 8 + 13 + 10 + 58?'), prefix,
                                   0.0, 16, seed=0)
    tail = full[len(prefix):]
    return '150' in tail[:24], tail[:80]
