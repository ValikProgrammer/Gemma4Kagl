#!/usr/bin/env python3
"""Build notebooks/smoke/ (a Kaggle notebook + kernel-metadata.json) from the official starter.

The notebook is the starter with these changes:
  1. our ./sample_submission files are embedded and written over the stock ones,
  2. it runs N tasks (one per repo first) or a fixed list from --task-file instead of tasks[:2],
  3. per task it saves diagnostics under results/ (see DIAG below): the full agent_patch,
     the tail of the verification output, the console transcript and one JSON row of metrics,
  4. EvalConfig(verbose=True) so results/logs/<id>.log is not empty (it was 0 bytes with False),
  5. a last cell lists the logs/traces and zips results/ into /kaggle/working for download.

    python3 -I scripts/build_smoke_notebook.py            # 3 tasks, 12b model
    python3 -I scripts/build_smoke_notebook.py --tasks 5 --model gemma-4-31b-it-qat-w4a16-ct
    python3 -I scripts/build_smoke_notebook.py --task-file smoke_tasks_30.txt --model gemma-4-31b-it-qat-w4a16-ct
Summarize a downloaded run: python3 -I scripts/summarize_run.py <extracted output dir>
Push (this starts a run and spends GPU quota):
    kaggle kernels push -p notebooks/smoke
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = ROOT / "notebooks" / "upstream_getting_started.ipynb"
SUBMISSION = ROOT / "sample_submission"
OUT = ROOT / "notebooks" / "smoke"
KERNEL_SLUG = "gemma4-smoke-test"
USERNAME = "valentinhilko"
DOCKER_IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
TASKS_JSONL = ROOT / "tasks.jsonl"

# Per-task diagnostics, defined right after `evaluator = Evaluator(eval_config)`.
# Writes under results/ (all of it goes into smoke_logs.zip):
#   patches/<id>.diff        result.agent_patch exactly as returned (not stripped, not cut)
#   test_outputs/<id>.txt    last TEST_TAIL_LINES lines of every text field of the result that looks
#                            like verification output (the field names are not documented)
#   console/<id>.txt         everything evaluate_task printed to stdout (Rich transcript with verbose)
#   smoke_results.jsonl      one row per task: scalar result fields, lengths of long ones, wall time,
#                            and `_fields` (name -> type[len]) so we learn the result object's shape
DIAG = r"""
# --- smoke diagnostics (added by scripts/build_smoke_notebook.py) ---
import contextlib, dataclasses, inspect, json as _json, time as _time
RES_DIR = WORKING_DIR / 'results'
for _d in ('patches', 'test_outputs', 'console'):
    (RES_DIR / _d).mkdir(parents=True, exist_ok=True)
TEST_TAIL_LINES = 200
_TEST_HINTS = ('test', 'pytest', 'verif', 'stdout', 'stderr', 'output', 'log', 'error')
_EXTRA_ATTRS = ('test_output', 'test_stdout', 'test_stderr', 'verification_output', 'verification_stdout',
                'verification_stderr', 'pytest_output', 'test_log', 'stdout', 'stderr', 'output', 'error',
                'agent_error', 'submitted_patch', 'patch_submitted')

def _as_dict(obj):
    for m in ('model_dump', 'dict'):  # pydantic v2 / v1
        if callable(getattr(obj, m, None)):
            try:
                return dict(getattr(obj, m)())
            except Exception:
                pass
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    return dict(getattr(obj, '__dict__', {}))

def _flatten(d, prefix=''):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _flatten(v, f'{prefix}{k}.')
        else:
            yield f'{prefix}{k}', v

def _save_diag(task, result, wall_s):
    iid = task.instance_id
    raw = result.agent_patch or ''
    (RES_DIR / 'patches' / f'{iid}.diff').write_text(raw, encoding='utf-8')
    fields = dict(_flatten(_as_dict(result)))
    for a in _EXTRA_ATTRS:  # properties do not show up in model_dump/__dict__
        if a not in fields:
            try:
                v = getattr(result, a, None)
            except Exception:
                v = None
            if v is not None:
                fields[a] = v
    row = {'id': iid, 'repo': task.repo, 'wall_s': round(wall_s, 1),
           'agent_patch_chars_raw': len(raw), 'agent_patch_ends_with_newline': raw.endswith('\n')}
    shapes, tails = {}, []
    for name, v in fields.items():
        shapes[name] = type(v).__name__ + (f'[{len(v)}]' if isinstance(v, (str, list, tuple, dict)) else '')
        if v is None or isinstance(v, (bool, int, float)):
            row[name] = v
        elif isinstance(v, str):
            if len(v) <= 300:
                row[name] = v
            else:
                row[name + '_chars'] = len(v)
            if (name not in ('agent_patch', 'submitted_patch') and v.strip()
                    and any(h in name.lower() for h in _TEST_HINTS)):
                tails.append((name, v))
        elif isinstance(v, (list, tuple)):
            row[name + '_len'] = len(v)
        else:
            row[name] = str(v)[:300]
    row['_fields'] = shapes
    out = []
    for name, v in tails:
        lines = v.splitlines()
        out.append(f'===== result.{name}: last {min(len(lines), TEST_TAIL_LINES)} of {len(lines)} lines =====')
        out.extend(lines[-TEST_TAIL_LINES:])
    if not out:
        out = ['no verification output found on the result object; fields: ' + ', '.join(sorted(shapes))]
    (RES_DIR / 'test_outputs' / f'{iid}.txt').write_text('\n'.join(out) + '\n', encoding='utf-8')
    with open(RES_DIR / 'smoke_results.jsonl', 'a', encoding='utf-8') as fh:
        fh.write(_json.dumps(row, default=str) + '\n')
    if len(raw) in (8191, 8192, 8193):
        print(f'  !! agent_patch is {len(raw)} chars: probably truncated inside the harness')

async def _evaluate_logged(task, task_index, total_tasks):
    # Capture what evaluate_task prints (Rich transcript when verbose=True) into console/<id>.txt
    with open(RES_DIR / 'console' / f'{task.instance_id}.txt', 'w', encoding='utf-8') as fh, \
            contextlib.redirect_stdout(fh):
        r = evaluator.evaluate_task(task=task, task_index=task_index, total_tasks=total_tasks)
        if inspect.isawaitable(r):
            r = await r
        return r
# --- end smoke diagnostics ---
"""


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=3, help="number of tasks (one per repo first, then fill up)")
    ap.add_argument("--task-file", type=Path, help="text file with one instance_id per line ('#' comments); overrides --tasks")
    ap.add_argument("--model", default="gemma-4-12b-it-qat-w4a16-ct")
    ap.add_argument("--max-minutes", type=float, default=10.0, help="per-task time cap for the smoke test")
    ap.add_argument("--no-verbose", dest="verbose", action="store_false",
                    help="keep the starter's EvalConfig(verbose=False); results/logs/*.log stay empty")
    args = ap.parse_args()

    task_ids = None
    if args.task_file:
        task_ids = [
            line.split("#", 1)[0].strip()
            for line in args.task_file.read_text(encoding="utf-8").splitlines()
        ]
        task_ids = [t for t in task_ids if t]
        known = {json.loads(line)["instance_id"] for line in TASKS_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()}
        unknown = [t for t in task_ids if t not in known]
        dupes = sorted({t for t in task_ids if task_ids.count(t) > 1})
        if not task_ids or unknown or dupes:
            raise SystemExit(f"bad --task-file {args.task_file}: empty={not task_ids} unknown={unknown} duplicates={dupes}")

    nb = json.loads(UPSTREAM.read_text())
    cells = nb["cells"]

    # Cell 4: model choice.
    src = "".join(cells[4]["source"])
    old = "MODEL_NAME = 'gemma-4-12b-it-qat-w4a16-ct'"
    assert old in src
    cells[4]["source"] = src.replace(old, f"MODEL_NAME = '{args.model}'").splitlines(True)

    # Embed our submission files and write them over the stock sample_submission.
    files = {
        p.relative_to(SUBMISSION).as_posix(): p.read_text(encoding="utf-8")
        for p in sorted(SUBMISSION.rglob("*"))
        if p.is_file() and p.suffix in {".yaml", ".yml", ".md", ".txt"}
    }
    overlay = (
        "# Overlay: our submission files replace the stock sample_submission ones\n"
        "# (generated by scripts/build_smoke_notebook.py from ./sample_submission)\n"
        f"OUR_FILES = {json.dumps(files, indent=1, ensure_ascii=False)}\n"
        "for rel, text in OUR_FILES.items():\n"
        "    p = AGENT_DIR / rel\n"
        "    p.parent.mkdir(parents=True, exist_ok=True)\n"
        "    if rel.endswith(('.yaml', '.yml')):  # keep the model in sync with MODEL_NAME\n"
        "        text = re.sub(r'^(\\s*model:\\s*).*$', rf'\\g<1>{MODEL_NAME}', text, flags=re.MULTILINE)\n"
        "    p.write_text(text, encoding='utf-8')\n"
        "# Drop the dummy 31B LoRA adapters (our YAMLs no longer reference them).\n"
        "shutil.rmtree(AGENT_DIR / 'adapters', ignore_errors=True)\n"
        "print('overlaid', len(OUR_FILES), 'files:', sorted(OUR_FILES))\n"
    )
    cells.insert(5, code(overlay))  # right after the cell that loads tasks

    # Task selection + smoke-test time cap (cells shifted by one).
    for c in cells:
        s = "".join(c["source"])
        if "SAMPLE_TASKS = tasks[:2]" in s:
            if task_ids:
                pick = (
                    f"# Fixed task list from {args.task_file.name} (scripts/build_smoke_notebook.py --task-file)\n"
                    f"TASK_IDS = {json.dumps(task_ids)}\n"
                    "_by_id = {t.instance_id: t for t in tasks}\n"
                    "_missing = [i for i in TASK_IDS if i not in _by_id]\n"
                    "assert not _missing, f'task ids not in tasks.jsonl: {_missing}'\n"
                    "SAMPLE_TASKS = [_by_id[i] for i in TASK_IDS]\n"
                    f"max_time_minutes = min(max_time_minutes, {args.max_minutes})  # smoke-test cap\n"
                    "print('tasks:', [t.instance_id for t in SAMPLE_TASKS])"
                )
                s = s.replace("SAMPLE_TASKS = tasks[:2]", pick)
            else:
                pick = (
                "# One task per repo first, then fill up to N\n"
                "_seen, SAMPLE_TASKS = set(), []\n"
                "for t in tasks:\n"
                "    if t.repo not in _seen:\n"
                "        _seen.add(t.repo); SAMPLE_TASKS.append(t)\n"
                f"for t in tasks:\n"
                f"    if len(SAMPLE_TASKS) >= {args.tasks}: break\n"
                "    if t not in SAMPLE_TASKS: SAMPLE_TASKS.append(t)\n"
                f"SAMPLE_TASKS = SAMPLE_TASKS[:{args.tasks}]\n"
                f"max_time_minutes = min(max_time_minutes, {args.max_minutes})  # smoke-test cap\n"
                "print('tasks:', [t.instance_id for t in SAMPLE_TASKS])"
                )
                s = s.replace("SAMPLE_TASKS = tasks[:2]", pick)
            # Diagnostics: hook into the evaluation loop (each anchor must exist exactly once).
            for old, new in [
                ("evaluator = Evaluator(eval_config)\n", "evaluator = Evaluator(eval_config)\n" + DIAG),
                ("    result = run_sync(\n        evaluator.evaluate_task,",
                 "    _t0 = _time.time()\n    result = run_sync(\n        _evaluate_logged,"),
                ("    patch = (result.agent_patch or '').strip()\n",
                 "    patch = (result.agent_patch or '').strip()\n"
                 "    try:\n        _save_diag(task, result, _time.time() - _t0)\n"
                 "    except Exception as e:  # diagnostics must never stop the run\n"
                 "        print('  diag failed:', repr(e))\n"),
                ("    verbose=False,\n", f"    verbose={args.verbose},\n"),
            ]:
                if s.count(old) != 1:
                    raise SystemExit(f"anchor not found once in upstream eval cell: {old!r}")
                s = s.replace(old, new)
            c["source"] = s.splitlines(True)
            break
    else:
        raise SystemExit("SAMPLE_TASKS line not found in upstream notebook")

    # Final cell: surface logs.
    cells.append(md("## Logs\nEverything under `/kaggle/working/results` plus `smoke_logs.zip` appears in the notebook Output tab."))
    cells.append(code(
        "import shutil, subprocess\n"
        "res = WORKING_DIR / 'results'\n"
        "print(subprocess.run(['find', str(res), '-maxdepth', '2', '-type', 'f'], capture_output=True, text=True).stdout)\n"
        "print(subprocess.run(['du', '-sh', *sorted(str(p) for p in res.iterdir())], capture_output=True, text=True).stdout)\n"
        "summary = res / 'summary.json'\n"
        "print(summary.read_text() if summary.exists() else 'no summary.json')\n"
        "shutil.make_archive(str(WORKING_DIR / 'smoke_logs'), 'zip', root_dir=res)\n"
    ))

    OUT.mkdir(parents=True, exist_ok=True)
    nb_path = OUT / "smoke.ipynb"
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))

    meta = json.loads(json.dumps({
        "id": f"{USERNAME}/{KERNEL_SLUG}",
        "title": "Gemma4 smoke test",
        "code_file": "smoke.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": ["metric/gemma-4-developer-agent-wheelhouse"],
        "kernel_sources": [],
        "competition_sources": ["gemma-4-developer-agent"],
        "model_sources": [f"google/gemma-4/Other/{args.model}/2"],
        "machine_shape": "NvidiaL4",
        # Pinned image from the official starter: the wheelhouse is built for Python 3.12,
        # the default Kaggle image has 3.13 and pip fails with "not a supported wheel".
        "docker_image": DOCKER_IMAGE,
    }))
    (OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    n = f"{len(task_ids)} tasks from {args.task_file}" if task_ids else f"{args.tasks} tasks"
    print(f"wrote {nb_path} ({len(files)} embedded files, {n}, model {args.model}, "
          f"max {args.max_minutes} min/task, verbose={args.verbose})")


if __name__ == "__main__":
    main()
