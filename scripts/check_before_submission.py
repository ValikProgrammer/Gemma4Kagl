#!/usr/bin/env python3
"""Offline sanity check for a swegemma submission directory, plus zip packaging.

Mirrors the rules in docs/HARNESS_README.md (section 2-3) so a broken submission
is caught before it costs a daily Kaggle submission.

    python3 -I scripts/check_before_submission.py sample_submission
    python3 -I scripts/check_before_submission.py sample_submission --zip submission.zip
"""
import argparse
import re
import sys
import zipfile
from pathlib import Path

import yaml

ALLOWED_MODELS = {
    "gemma-4-31b-it-qat-w4a16-ct",
    "gemma-4-12b-it-qat-w4a16-ct",
    "gemma-4-e4b-it-qat-w4a16-ct",
    "gemma-4-e2b-it-qat-w4a16-ct",
}
TOOLS = {
    "run_command", "submit_patch", "get_status", "read_file", "edit_file",
    "write_file", "get_code_neighbors", "search_similar_code", "get_code_subgraph",
}
ROOT_NAMES = ["agent.yaml", "agent.yml", "root_agent.yaml", "root_agent.yml"]
ALLOWED_EXT = {".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"}
MAX_BYTES = 3 * 1024**3
MAX_TOKENS = 32768
ADAPTER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*$")
EVAL_KEYS = {"timeout_seconds", "max_tool_calls", "max_time_minutes", "max_turns"}

errors: list[str] = []
warnings: list[str] = []


def err(msg):
    errors.append(msg)


def warn(msg):
    warnings.append(msg)


class IncludeError(Exception):
    pass


def load_yaml(path: Path, root: Path, depth=0, seen=()):
    """Load YAML, resolving `!include` the way the harness does (sandboxed)."""
    if depth > 10:
        raise IncludeError(f"{path}: include depth > 10")
    if path in seen:
        raise IncludeError(f"{path}: include cycle")

    class Loader(yaml.SafeLoader):
        pass

    def include(loader, node):
        rel = loader.construct_scalar(node)
        if rel.startswith("/") or "\0" in rel:
            raise IncludeError(f"{path}: illegal include path {rel!r}")
        target = path.parent / rel
        if not target.is_file():
            raise IncludeError(f"{path.relative_to(root)}: !include {rel} -> file not found")
        if root.resolve() not in target.resolve().parents:
            raise IncludeError(f"{path.relative_to(root)}: !include {rel} escapes the submission root")
        if ".." in Path(rel).parts:
            # The starter kit itself uses `../prompts/...`, so the real loader accepts it
            # when the target stays inside the root, although the README says `..` is blocked.
            warn(f"{path.relative_to(root)}: !include {rel} uses `..` (starter kit does too; "
                 "if the validator rejects it, move the file next to the YAML)")
        if target.suffix in (".md", ".txt"):
            return target.read_text(encoding="utf-8")
        if target.suffix in (".yaml", ".yml"):
            return load_yaml(target, root, depth + 1, seen + (path,))
        raise IncludeError(f"{path}: cannot include {target.suffix} files")

    Loader.add_constructor("!include", include)
    return yaml.load(path.read_text(encoding="utf-8"), Loader=Loader)


def walk_agent(cfg_path: Path, root: Path, visited: dict, depth=0):
    """Collect agents (name, model, adapter, tools) reachable from a config file."""
    if cfg_path in visited:
        return
    rel = cfg_path.relative_to(root)
    try:
        cfg = load_yaml(cfg_path, root)
    except (IncludeError, yaml.YAMLError) as e:
        err(str(e))
        return
    if not isinstance(cfg, dict):
        err(f"{rel}: top level must be a mapping")
        return
    visited[cfg_path] = cfg
    agent_class = cfg.get("agent_class", "LlmAgent")
    name = cfg.get("name")
    if not name:
        err(f"{rel}: missing `name`")

    if agent_class == "LlmAgent":
        if not cfg.get("model"):
            err(f"{rel}: missing `model`")
        for t in cfg.get("tools") or []:
            if isinstance(t, str):
                if t not in TOOLS:
                    err(f"{rel}: unknown tool {t!r} (known: {', '.join(sorted(TOOLS))})")
            elif isinstance(t, dict) and "agent_tool" in t:
                cp = t["agent_tool"].get("config_path")
                if not cp:
                    err(f"{rel}: agent_tool without config_path")
                else:
                    child = (cfg_path.parent / cp)
                    if not child.is_file():
                        child = root / cp
                    if child.is_file():
                        walk_agent(child, root, visited, depth + 1)
                    else:
                        err(f"{rel}: agent_tool config_path {cp} not found")
            else:
                warn(f"{rel}: unrecognised tool entry {t!r}")
        gen = cfg.get("generate_content_config")
        if isinstance(gen, dict):
            check_generation(gen, rel)
    for sub in cfg.get("sub_agents") or []:
        cp = sub.get("config_path") if isinstance(sub, dict) else None
        if cp:
            child = cfg_path.parent / cp
            if not child.is_file():
                child = root / cp
            if child.is_file():
                walk_agent(child, root, visited, depth + 1)
            else:
                err(f"{rel}: sub_agent config_path {cp} not found")


def check_generation(gen, rel):
    for bad in ("tools", "system_instruction", "http_options", "safety_settings", "response_schema"):
        if bad in gen:
            err(f"{rel}: generate_content_config.{bad} is not allowed")
    mot = gen.get("max_output_tokens")
    if mot is not None and not 1 <= mot <= MAX_TOKENS:
        err(f"{rel}: max_output_tokens {mot} outside 1..{MAX_TOKENS}")
    tc = gen.get("thinking_config") or {}
    tb = tc.get("thinking_budget")
    if tb is not None and not 0 <= tb <= MAX_TOKENS:
        err(f"{rel}: thinking_budget {tb} outside 0..{MAX_TOKENS}")
    if "thinking_level" in tc:
        warn(f"{rel}: thinking_level maps to reasoning_effort on vLLM; "
             "the README recommends include_thoughts + thinking_budget instead")
    if (mot or 0) + (tb or 0) > MAX_TOKENS:
        warn(f"{rel}: max_output_tokens + thinking_budget exceeds {MAX_TOKENS}; context may overflow")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submission_dir", type=Path)
    ap.add_argument("--zip", type=Path, help="write submission.zip here if all checks pass")
    args = ap.parse_args()
    root = args.submission_dir.resolve()
    if not root.is_dir():
        sys.exit(f"{root} is not a directory")

    # 1. files: extensions, symlinks, size
    total = 0
    files = [p for p in root.rglob("*") if p.is_file() or p.is_symlink()]
    for p in files:
        rel = p.relative_to(root)
        if p.is_symlink():
            err(f"{rel}: symlinks are not allowed")
            continue
        total += p.stat().st_size
        if p.suffix not in ALLOWED_EXT:
            err(f"{rel}: extension {p.suffix or '(none)'} not allowed")
    if total > MAX_BYTES:
        err(f"total size {total / 1024**3:.2f} GiB exceeds 3 GiB")
    if len(files) > 10_000:
        err(f"{len(files)} files > 10,000")

    # 2. root config
    roots = [root / n for n in ROOT_NAMES if (root / n).is_file()]
    if len(roots) != 1:
        err(f"need exactly one root config, found {len(roots)}: {[r.name for r in roots]}")
        report(root, None)
        return

    # 3. agent tree
    visited: dict = {}
    walk_agent(roots[0], root, visited)

    models = {str(c.get("model", "")).split("/")[-1] for c in visited.values() if c.get("model")}
    if len(models) > 1:
        err(f"more than one base model declared: {sorted(models)}")
    for m in models:
        if m not in ALLOWED_MODELS:
            err(f"model {m!r} is not an allowed competition model")

    # 4. adapters
    wanted = {c["adapter"] for c in visited.values() if c.get("adapter")}
    adapters_dir = root / "adapters"
    have = {d.name for d in adapters_dir.iterdir() if d.is_dir()} if adapters_dir.is_dir() else set()
    for a in sorted(wanted):
        if not ADAPTER_NAME.match(a):
            err(f"adapter name {a!r} is invalid")
        elif a not in have:
            err(f"adapter {a!r} is referenced but adapters/{a}/ does not exist "
                "(remove the `adapter:` line or add the weights)")
        else:
            for f in ("adapter_config.json", "adapter_model.safetensors"):
                if not (adapters_dir / a / f).is_file():
                    err(f"adapters/{a}/{f} is missing")
    for a in sorted(have - wanted):
        warn(f"adapters/{a}/ is never used by any agent (it still counts toward the 3 GiB)")

    # 5. eval_config.yaml
    ev = root / "eval_config.yaml"
    if ev.is_file():
        try:
            cfg = yaml.safe_load(ev.read_text()) or {}
            e = cfg.get("evaluation") or {}
            for k in e:
                if k not in EVAL_KEYS:
                    err(f"eval_config.yaml: unknown key evaluation.{k}")
            if e.get("max_time_minutes", 60) * 120 > 24 * 60:
                warn(f"eval_config.yaml: max_time_minutes={e['max_time_minutes']} x ~120 tasks "
                     "is more than 24h worst case; check the notebook time limit")
        except yaml.YAMLError as exc:
            err(f"eval_config.yaml: {exc}")
    else:
        warn("no eval_config.yaml: harness defaults apply (100 calls, 60 min, 300 s)")

    report(root, visited)
    if args.zip and not errors:
        args.zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(files):
                z.write(p, p.relative_to(root).as_posix())
        print(f"wrote {args.zip} ({args.zip.stat().st_size / 1024:.1f} KiB), root files at archive top level")


def report(root, visited):
    if visited:
        print(f"agents ({len(visited)}):")
        for p, c in visited.items():
            tools = [t if isinstance(t, str) else "agent_tool" for t in c.get("tools") or []]
            print(f"  {p.relative_to(root)}: {c.get('name')}  model={c.get('model')}  "
                  f"adapter={c.get('adapter')}  tools={len(tools)}")
    for w in warnings:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")
    print("\nFAILED" if errors else "\nOK", f"({len(errors)} errors, {len(warnings)} warnings)")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
