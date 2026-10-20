# Harness notes

Condensed from `HARNESS_README.md`. Most failures come from the harness, not the model.

## Constraints
- One base model per submission: `gemma-4-{31b,12b,e4b,e2b}-it-qat-w4a16-ct`. LoRA adapters allowed.
- Submission is declarative YAML + prompts + `.safetensors`, under 3 GiB. No custom Python.
- vLLM `max_model_len=32768`: prompt + thinking + output share it. Default `thinking_budget` 4096, `max_output_tokens` 16384.
- Sandbox is offline, 4 GiB RAM, 2 vCPU. `run_command` output is cut at 5000 chars, `read_file` at 150 lines / 10k chars.

## Scoring
- Resolved only if pytest exits 0 and every FAIL_TO_PASS / PASS_TO_PASS test passed (none skipped).
- Patch is `git add -N . && git diff` against baseline. Edits to tests, `conftest.py`, `pytest.ini`,
  `pyproject.toml`, `tox.ini`, `setup.cfg` are stripped. Fix library code only.
- If the patch does not apply (4 fallback passes), the task scores 0.

## Where agents lose points
1. **Truncated tool calls**: long thinking + big `edit_file` hits `max_output_tokens`, the `<|tool_call>` tag is cut.
   Use small incremental edits and a modest thinking budget.
2. **Scratch files in `/workspace`** end up in the patch. Use `/tmp/repro.py`.
3. **Premature `submit_patch()`**: it ends the session as soon as the turn finishes. Call it last, after verifying.
4. **Context overflow**: compaction triggers at ~14k tokens. Delegate exploration to a read-only sub-agent
   (`agent_tool` with `skip_summarization: true`) to keep `read_file` output out of the coder's context.
5. **Tool budget**: only `submit_patch` and `get_status` are free. Defaults are tight (sample: 10 calls, 1 minute).
   Tune `eval_config.yaml`.
6. **Hallucinated tool names / string-typed args** are coerced or returned as errors, but each costs a turn.

## Local run
`swegemma eval --tasks tasks.jsonl --snapshots-dir snapshots --submission-dir sample_submission --results-dir results/run_01 --sandbox docker --task-id <id>`
Check `results/*/traces/` and `test_outputs/` to see why a task failed.
