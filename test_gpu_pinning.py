"""
Offline test for the v14.4 GPU-pinning fix.

WHY THIS EXISTS
---------------
A real OOM trace (2026-08-05, T4 x2, mixed preset) showed device_map="auto"
does not greedily fill GPU 0 before touching GPU 1 -- it shards EACH model's
layers across every visible GPU per accelerate's own balancing heuristic. The
first (~5 GB, 4-bit) model left GPU 1 with only ~7.4 GB free just from being
loaded; the second model's own auto-placement then collided with 4-bit
dequantization's transient scratch memory there, OOMing on every retry. See
Mas_solver.py's changelog and the loading code comments for the full trace.

Fix: assign each DISTINCT local_hf model its own GPU (device_map={"": i}),
round-robin, instead of letting accelerate decide. This test verifies the
ASSIGNMENT logic only -- it constructs UnifiedLLMClient objects, which are
lazy (no model load, no CUDA call) until the first call_model(), so it runs
with no GPU and no models downloaded.

Run:  python test_gpu_pinning.py
"""

import sys
import os
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

failures = []


def check(label, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not cond:
        failures.append(label)


def with_gpu_count(n):
    """Patch torch.cuda so Mas_solver's pipeline constructor sees n GPUs."""
    return mock.patch.multiple(
        "torch.cuda",
        is_available=mock.DEFAULT,
        device_count=mock.DEFAULT,
    )


def main():
    import torch
    import Mas_solver as M

    # --- 1. Two distinct local_hf models, 2 GPUs -> pinned to 0 and 1 -------
    with mock.patch.object(torch.cuda, "is_available", return_value=True), \
         mock.patch.object(torch.cuda, "device_count", return_value=2):
        pipe = M.QualityAwarePipeline(heterogeneous_preset="qwen_math7b_mixed",
                                      use_cache=False)
        assigned = pipe._local_hf_device_assignment
        check("2 GPUs: exactly 2 distinct local_hf models pinned",
              len(assigned) == 2, f"got {assigned}")
        indices = sorted(assigned.values())
        check("2 GPUs: assigned indices are {0, 1}", indices == [0, 1],
              f"got {indices}")

        math_client = pipe.solver._get_client(M.AgentRole.MATHEMATICIAN)
        base_client = pipe.solver._get_client(M.AgentRole.BASELINE)
        check("Mathematician and Baseline use DIFFERENT models (mixed preset)",
              math_client.model_name != base_client.model_name,
              f"{math_client.model_name} vs {base_client.model_name}")
        check("Mathematician client got a device_index",
              math_client.device_index is not None)
        check("Baseline client got a device_index",
              base_client.device_index is not None)
        check("Mathematician and Baseline pinned to DIFFERENT GPUs",
              math_client.device_index != base_client.device_index,
              f"math={math_client.device_index} base={base_client.device_index}")

        # Same-model roles must share the cached client (and therefore GPU).
        hg_client = pipe.solver._get_client(M.AgentRole.HYPOTHESIS_GENERATOR)
        check("Same-model roles share one client (cache hit)",
              hg_client is math_client)

    # --- 2. Single GPU -> no pinning, old "auto" behaviour preserved --------
    with mock.patch.object(torch.cuda, "is_available", return_value=True), \
         mock.patch.object(torch.cuda, "device_count", return_value=1):
        pipe1 = M.QualityAwarePipeline(heterogeneous_preset="qwen_math7b_mixed",
                                       use_cache=False)
        check("1 GPU: no device pinning attempted",
              len(pipe1._local_hf_device_assignment) == 0,
              f"got {pipe1._local_hf_device_assignment}")
        math_client1 = pipe1.solver._get_client(M.AgentRole.MATHEMATICIAN)
        check("1 GPU: device_index stays None (falls back to device_map='auto')",
              math_client1.device_index is None)

    # --- 3. No GPU at all -> no pinning, no crash ----------------------------
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        pipe0 = M.QualityAwarePipeline(heterogeneous_preset="qwen_math7b_mixed",
                                       use_cache=False)
        check("no GPU: no device pinning attempted, no crash",
              len(pipe0._local_hf_device_assignment) == 0)

    # --- 4. Homogeneous preset (ONE local_hf model, 2 GPUs available) -------
    # Only one distinct model -> only cuda:0 is ever assigned. Confirms this
    # doesn't force multi-GPU sharding onto a preset that never needed it.
    with mock.patch.object(torch.cuda, "is_available", return_value=True), \
         mock.patch.object(torch.cuda, "device_count", return_value=2):
        pipeh = M.QualityAwarePipeline(heterogeneous_preset="qwen_math_7b_local",
                                       use_cache=False)
        assigned_h = pipeh._local_hf_device_assignment
        check("homogeneous preset: exactly 1 distinct local_hf model",
              len(assigned_h) == 1, f"got {assigned_h}")
        check("homogeneous preset: that model pinned to cuda:0",
              list(assigned_h.values()) == [0], f"got {assigned_h}")

    # --- 5. Direct UnifiedLLMClient construction defaults to no pinning -----
    c = M.UnifiedLLMClient(provider="local_hf", model_override="x/y")
    check("bare UnifiedLLMClient: device_index defaults to None",
          c.device_index is None)
    c2 = M.UnifiedLLMClient(provider="local_hf", model_override="x/y", device_index=1)
    check("bare UnifiedLLMClient: device_index is settable directly",
          c2.device_index == 1)


if __name__ == "__main__":
    print("=" * 70)
    print("v14.4 — LOCAL_HF GPU PINNING (offline, no CUDA, no models)")
    print("=" * 70)
    main()
    print("\n" + "=" * 70)
    if failures:
        print(f"  {len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print(f"    - {f}")
        sys.exit(1)
    print("  ALL CHECKS PASSED")
    print("=" * 70)
