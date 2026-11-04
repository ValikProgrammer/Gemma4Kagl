#!/usr/bin/env python3
"""Summarize one smoke-test run as a markdown table (per task + totals + time projection).

    python3 -I scripts/summarize_run.py <extracted run output dir> [--tasks 116] [--limit-hours 12]

The folder is what the Kaggle Output tab gives (smoke_logs.zip extracted into results/ or not).
Everything is read as data with json/re; nothing from the folder is imported or executed.
Sources, all optional, merged per instance_id:
  - kernel log (<slug>.log, Kaggle JSON stream list, or plain text): the "-> resolved=..." lines,
    harness warnings (timeouts, no submit), setup time before the first task;
  - **/smoke_results.jsonl (written by the notebook since the diagnostics change);
  - **/trace_<id>.json (ATIF traces): tool calls, LLM calls, tokens, compactions;
  - **/patches/<id>.diff: full patch length.
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

RESULT_RE = re.compile(
    r"-> resolved=(?P<resolved>\w+), exit_code=(?P<exit>-?\d+|None), patch_chars=(?P<patch>\d+), "
    r"tool_calls=(?P<tools>\d+), duration=(?P<dur>[\d.]+)s"
)
START_RE = re.compile(r"\[(?P<i>\d+)/(?P<n>\d+)\] Evaluating (?P<id>\S+) ")
WARN_RE = re.compile(r"Task (?P<id>\S+) (?P<what>agent timed out|completed without explicit submit_patch call)")
SHORT = {
    "run_command": "rc", "read_file": "rf", "edit_file": "ef", "write_file": "wf", "submit_patch": "sp",
    "get_status": "gs", "search_similar_code": "ssc", "get_code_neighbors": "gcn",
    "get_code_subgraph": "gcs", "code_analyzer_agent": "sub",
}
COMPACTION_DROP = 0.7  # prompt tokens fall below 70% of the previous call of the same agent


def kernel_lines(path):
    """Yield (time_s or None, text line) from a Kaggle kernel log or a plain text log."""
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        records = json.loads(text)
    except ValueError:
        records = None
    if isinstance(records, list):
        for r in records:
            if isinstance(r, dict):
                for line in str(r.get("data", "")).splitlines():
                    yield r.get("time"), line
    else:
        for line in text.splitlines():
            yield None, line


def parse_kernel_log(root, rows, order, meta):
    logs = [p for p in sorted(root.glob("*.log")) if p.is_file() and p.stat().st_size > 0]
    for path in logs:
        current, seen_any = None, False
        for t, line in kernel_lines(path):
            m = START_RE.search(line)
            if m:
                current, seen_any = m["id"], True
                if current not in order:
                    order.append(current)
                if m["i"] == "1" and t is not None:
                    meta["setup_s"] = t
                if t is not None:
                    rows[current]["_start_t"] = t
                continue
            m = WARN_RE.search(line)
            if m:
                key = "timed_out" if "timed out" in m["what"] else "no_submit_no_changes"
                rows[m["id"]][key] = True
                continue
            m = RESULT_RE.search(line)
            if m and current:
                r = rows[current]
                r["resolved"] = m["resolved"] == "True"
                r["exit"] = None if m["exit"] == "None" else int(m["exit"])
                r["patch_chars"] = int(m["patch"])
                r["tool_calls_harness"] = int(m["tools"])
                r["wall_s"] = float(m["dur"])
                if t is not None and "_start_t" in r:
                    r["wall_s_kernel"] = t - r["_start_t"]
                if t is not None:
                    meta["last_task_end_s"] = t
        if seen_any:
            meta["kernel_log"] = path.name


def parse_results_jsonl(root, rows, order):
    for path in sorted(root.rglob("smoke_results.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            iid = d.get("id")
            if not iid:
                continue
            if iid not in order:
                order.append(iid)
            r = rows[iid]
            if "resolved" in d:
                r["resolved"] = bool(d["resolved"])
            if "test_exit_code" in d:
                r["exit"] = d["test_exit_code"]
            if "duration_seconds" in d:
                r["wall_s"] = d["duration_seconds"]
            elif "wall_s" in d:
                r["wall_s"] = d["wall_s"]
            if "agent_patch_chars_raw" in d:
                r["patch_chars_raw"] = d["agent_patch_chars_raw"]
            if "patch_submitted" in d:
                r["patch_submitted"] = d["patch_submitted"]


def parse_trace(path):
    d = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    tools = Counter()
    llm_calls = peak_prompt = max_out = compactions = 0
    last_prompt = {}
    nudges = 0
    for s in d.get("steps", []):
        extra = s.get("extra") or {}
        if extra.get("event_type") == "continuation_nudge":
            nudges += 1
        for tc in s.get("tool_calls") or []:
            tools[tc.get("function_name", "?")] += 1
        m = s.get("metrics")
        if not m:
            continue
        llm_calls += int(s.get("llm_call_count") or 1)
        p, c = int(m.get("prompt_tokens") or 0), int(m.get("completion_tokens") or 0)
        peak_prompt, max_out = max(peak_prompt, p), max(max_out, c)
        author = extra.get("author")
        prev = last_prompt.get(author)
        if prev and p < COMPACTION_DROP * prev:
            compactions += 1
        last_prompt[author] = p
    steps = d.get("steps") or [{}]
    return {
        "tools": tools, "llm_calls": llm_calls, "peak_prompt": peak_prompt, "max_out": max_out,
        "compactions": compactions, "nudges": nudges, "submit_called": tools.get("submit_patch", 0) > 0,
        "agent_s": (steps[-1].get("extra") or {}).get("elapsed_s"),
        "total_prompt": (d.get("final_metrics") or {}).get("total_prompt_tokens"),
        "total_completion": (d.get("final_metrics") or {}).get("total_completion_tokens"),
    }


def fmt(v, nd=0):
    if v is None:
        return "?"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--tasks", type=int, default=116, help="task count for the time projection")
    ap.add_argument("--limit-hours", type=float, default=12.0, help="run time limit to compare with")
    args = ap.parse_args()
    root = args.run_dir
    if not root.is_dir():
        sys.exit(f"not a directory: {root}")

    rows, order, meta = defaultdict(dict), [], {}
    parse_kernel_log(root, rows, order, meta)
    parse_results_jsonl(root, rows, order)
    for path in sorted(root.rglob("trace_*.json")):
        iid = path.stem[len("trace_"):]
        try:
            rows[iid].update(parse_trace(path))
        except (ValueError, OSError) as e:
            print(f"warning: cannot read {path}: {e}", file=sys.stderr)
            continue
        if iid not in order:
            order.append(iid)
    for path in sorted(root.rglob("patches/*.diff")):
        rows[path.stem]["patch_chars_raw"] = len(path.read_text(encoding="utf-8", errors="replace"))
    if not order:
        sys.exit(f"no kernel log, smoke_results.jsonl or traces found under {root}")

    print(f"# Run summary: {root}\n")
    src = [f"kernel log `{meta['kernel_log']}`" if "kernel_log" in meta else "no kernel log"]
    print("Sources: " + ", ".join(src) + f"; {sum('tools' in rows[i] for i in order)} traces.\n")
    print("| task | resolved | exit | wall s | agent s | tool calls by tool | LLM calls | compactions* "
          "| peak prompt tok | max out tok/step | patch chars | submit_patch | notes |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    all_tools = Counter()
    for iid in order:
        r = rows[iid]
        tools = r.get("tools") or Counter()
        all_tools.update(tools)
        by_tool = " ".join(f"{SHORT.get(k, k)}{v}" for k, v in tools.most_common())
        n_counted = sum(v for k, v in tools.items() if k not in ("submit_patch", "get_status"))
        patch = r.get("patch_chars_raw", r.get("patch_chars"))
        notes = []
        if r.get("timed_out"):
            notes.append("timeout")
        if r.get("no_submit_no_changes"):
            notes.append("no changes")
        if r.get("nudges"):
            notes.append(f"nudges {r['nudges']}")
        if patch in (8191, 8192, 8193):
            notes.append("patch cut at 8192?")
        if "tool_calls_harness" in r and r["tool_calls_harness"] != n_counted and tools:
            notes.append(f"harness counted {r['tool_calls_harness']}")
        print(f"| {iid} | {fmt(r.get('resolved'))} | {fmt(r.get('exit'))} | {fmt(r.get('wall_s'))} "
              f"| {fmt(r.get('agent_s'))} | {n_counted}: {by_tool} | {fmt(r.get('llm_calls'))} "
              f"| {fmt(r.get('compactions'))} | {fmt(r.get('peak_prompt'))} | {fmt(r.get('max_out'))} "
              f"| {fmt(patch)} | {fmt(r.get('submit_called'))} | {', '.join(notes)} |")

    n = len(order)
    resolved = sum(bool(rows[i].get("resolved")) for i in order)
    walls = [rows[i]["wall_s"] for i in order if rows[i].get("wall_s") is not None]
    mean_wall = sum(walls) / len(walls) if walls else None

    def total(key):
        return sum(rows[i].get(key) or 0 for i in order)

    print("\n**Totals**\n")
    print(f"- resolved: {resolved}/{n} ({100 * resolved / n:.0f}%)")
    print(f"- exit codes: {dict(Counter(fmt(rows[i].get('exit')) for i in order))}")
    print(f"- tool calls: {sum(all_tools.values())} ({', '.join(f'{k} {v}' for k, v in all_tools.most_common())})")
    print(f"- LLM calls: {total('llm_calls')}; compactions*: {total('compactions')}; "
          f"timeouts: {sum(bool(rows[i].get('timed_out')) for i in order)}; "
          f"submit_patch called: {sum(bool(rows[i].get('submit_called')) for i in order)}/{n}")
    print(f"- tokens: prompt {total('total_prompt')}, completion {total('total_completion')}; "
          f"peak prompt {max((rows[i].get('peak_prompt') or 0) for i in order)}, "
          f"max output/step {max((rows[i].get('max_out') or 0) for i in order)}")
    if mean_wall is not None:
        setup = meta.get("setup_s")
        proj_h = (args.tasks * mean_wall + (setup or 0)) / 3600
        fit = "fits" if proj_h <= args.limit_hours else "DOES NOT fit"
        print(f"- wall time: {sum(walls) / 60:.1f} min for {len(walls)} tasks, mean {mean_wall:.0f} s/task, "
              f"max {max(walls):.0f} s")
        print(f"- setup before first task: {fmt(setup)} s" + (" (wheels + vLLM start)" if setup else ""))
        print(f"- projected for {args.tasks} tasks: {args.tasks} x {mean_wall:.0f} s"
              + (f" + {setup:.0f} s setup" if setup else "")
              + f" = {proj_h:.1f} h ({fit} in {args.limit_hours:g} h; same caps and sequential run assumed)")
    print("\n\\* compactions are inferred: an LLM call whose prompt is < "
          f"{COMPACTION_DROP:.0%} of the previous call of the same agent. "
          "Tool abbreviations: " + ", ".join(f"{v}={k}" for k, v in SHORT.items()) + ".")


if __name__ == "__main__":
    main()
