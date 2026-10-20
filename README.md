# Gemma4Kagl

Submission for the Kaggle "Gemma 4 Developer Agent" competition: a declarative
agent (YAML + prompts + optional LoRA) that fixes SWE-bench-style issues in
`fastapi`, `rich`, `requests` and `httpx`, scored by resolution rate.

- `tasks.jsonl` - 129 public training tasks (fastapi 67, rich 48, requests 13, httpx 1)
- `docs/HARNESS_README.md` - official `swegemma` harness reference
- `docs/harness-notes.md` - our condensed notes: what the harness does and where agents lose points

Large data (`snapshots/`, `docker/`, `wheels/`, ...) is downloaded locally and git-ignored.
