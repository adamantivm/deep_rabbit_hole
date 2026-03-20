# Results: Idea D Real-ONNX MCTS Parity Test

## Summary

Implemented Idea D from the debugging plan: a deterministic MCTS parity test that compares Rust vs
Python behaviour using the same real ONNX model weights. Both implementations run identical MCTS
search and their step-by-step game traces are compared field by field.

**Outcome: PASSED ✅** — all structural fields match exactly; root-value and policy floats match
within epsilon 1e-5.

---

## What Was Implemented

### Python side — `deep_quoridor/src/mcts_game_reference.py`

- Added optional `--onnx-model <path>` CLI flag (onnx_model is `None` by default → existing
  `UniformMockNNEvaluator` behaviour unchanged).
- Added `OnnxNNEvaluator` class:
  - Loads the ONNX session via `onnxruntime.InferenceSession`.
  - Calls the model with the same 5-channel ResNet input the Python trainer uses.
  - Applies masked softmax (matching Rust `evaluator.rs:masked_softmax()`).
  - Implements `evaluate_batch()` API accepted by `MCTS.search()`.
- `create_runner()` selects evaluator based on the optional path argument.
- `FileNotFoundError` raised immediately if the model path does not exist.

### Rust side — `deep_quoridor/rust/src/python_consistency.rs`

- Added `resolve_idea_d_onnx_model_path()`:
  - Reads `DEEP_QUORIDOR_ONNX_MODEL` env var if set, otherwise defaults to
    `rust/fixtures/alphazero_B5W2_mv1.0_v509.onnx`.
  - Hard-fails with a clear actionable message if the resolved path does not exist.
- Added `run_mcts_game_python_with_model()`:
  - Wraps existing `run_mcts_game_python()` with an optional model path forwarded as
    `--onnx-model` to the Python script.
- Added `hex_to_f32_vec()` and `assert_f32_vec_close()` helpers for element-wise epsilon comparison
  of float32 arrays encoded as hex strings.
- Updated `assert_snapshot_fields_match()`:
  - Root value: epsilon comparison (`1e-5`) instead of exact hex equality.
  - Root policy: element-wise epsilon comparison (`1e-5`) instead of exact hex equality.
  - All structural fields (G, P, W, C, M, T, RM, RT, A) still require **exact** equality.
- Added `test_mcts_game_trace_matches_python_onnx` test (board B5W2, mcts_n=20, max_steps=50):
  - Calls Python trace script with ONNX evaluator.
  - Runs Rust MCTS with `OnnxEvaluator` loaded from the same model file.
  - Compares step-by-step: exact structural fields, epsilon floats, same selected action.
  - On failure: dumps both traces to `/tmp` and auto-explains via `--explain-trace`.

---

## Test Commands

```bash
# Run new ONNX parity test only
cd deep_quoridor/rust
PYTHON=/path/to/.venv/bin/python cargo test --features binary test_mcts_game_trace_matches_python_onnx -- --nocapture

# Run both parity tests (mock + ONNX)
PYTHON=/path/to/.venv/bin/python cargo test --features binary test_mcts_game_trace_matches_python -- --nocapture

# Use a different ONNX model
DEEP_QUORIDOR_ONNX_MODEL=/path/to/other.onnx cargo test --features binary test_mcts_game_trace_matches_python_onnx -- --nocapture
```

---

## Comparison Policy

| Field | Check |
|-------|-------|
| Grid (G) | Exact hex equality |
| Player positions (P) | Exact |
| Walls remaining (W) | Exact |
| Current player (C) | Exact |
| Action mask (M) | Exact |
| ResNet tensor (T) | Exact hex equality |
| Rotated mask (RM) | Exact (when player 1) |
| Rotated tensor (RT) | Exact (when player 1) |
| Root value (V) | Epsilon ≤ 1e-5 |
| Root policy (Q) | Element-wise epsilon ≤ 1e-5 |
| Selected action (A) | Exact index equality |

---

## Observed Results

- Both traces agree at every step for B5W2 with mcts_n=20, max_steps=50.
- No divergence detected in root-value, policy, or selected-action under epsilon 1e-5.
- Runtime: ~5s per test run (well under the 30s target).
- Regression check: existing `test_mcts_game_trace_matches_python` (mock evaluator) still passes.

---

## Files Changed

| File | Change |
|------|--------|
| `src/mcts_game_reference.py` | Added `OnnxNNEvaluator`, `--onnx-model` CLI flag |
| `rust/src/python_consistency.rs` | Added helpers, model resolver, ONNX parity test |

---

## Interpretation

The fact that both implementations agree with a real NN (not just a uniform mock) means:

1. The tensor construction, masking, and softmax are numerically equivalent.
2. MCTS tree search produces the same visit distributions under identical NN weights.
3. The selected action matches at every step.

This **does not rule out** bugs in replay-buffer value backfill, NPZ field ordering, or the trainer
sampling path (Idea C). Those remain as the next debugging priority.
