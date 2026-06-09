#!/usr/bin/env python3
"""Trial Tree State Manager for iterative CPU kernel optimization.

Manages a tree of optimization trials for each kernel, tracking parent-child
relationships, strategies, correctness, and speedup results. Supports
branching back to the best ancestor when a trial regresses.

Usage:
    python scripts/trial_manager.py init <kernel_name> <baseline_file>
    python scripts/trial_manager.py save <kernel_name> <trial_dir> --parent <parent_id> --strategy "description"
    python scripts/trial_manager.py result <kernel_name> <trial_id> --correctness <pass|fail> --speedup <float> --baseline_us <float> --kernel_us <float>
    python scripts/trial_manager.py status <kernel_name>
    python scripts/trial_manager.py best <kernel_name>
    python scripts/trial_manager.py baseline-us <kernel_name>
    python scripts/trial_manager.py finalize <kernel_name> <output_dir>
"""

import argparse
import json
import os
import shutil
import sys

TRIALS_DIR = os.path.join(os.getcwd(), "trials")
OUTPUT_DIR = os.path.join(os.getcwd(), "output")


def _state_path(kernel_name):
    return os.path.join(TRIALS_DIR, kernel_name, "state.json")


def _trial_dir(kernel_name):
    return os.path.join(TRIALS_DIR, kernel_name)


def _load_state(kernel_name):
    path = _state_path(kernel_name)
    if not os.path.exists(path):
        print(f"Error: No trial tree found for '{kernel_name}'. Run 'init' first.", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        state = json.load(f)
    return state


def _save_state(kernel_name, state):
    path = _state_path(kernel_name)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


# ============================================================================
# Commands
# ============================================================================


def cmd_init(args):
    """Initialize a new trial tree for a kernel."""
    kernel_name = args.kernel_name
    baseline_file = args.baseline_file

    trial_dir = _trial_dir(kernel_name)
    if os.path.exists(_state_path(kernel_name)):
        print(
            f"Warning: Trial tree for '{kernel_name}' already exists. "
            f"Use a different name or delete trials/{kernel_name}/."
        )
    else:
        os.makedirs(trial_dir, exist_ok=True)

        state = {
            "kernel_name": kernel_name,
            "baseline_file": baseline_file,
            "trials": {},
            "best_trial": None,
            "next_id": 0,
            "baseline_us": None,
        }
        _save_state(kernel_name, state)
        print(f"Initialized trial tree for '{kernel_name}' in trials/{kernel_name}/")
        print(f"  Baseline: {baseline_file}")


def cmd_save(args):
    """Save a trial by copying kernel files into the trial directory."""
    kernel_name = args.kernel_name
    trial_source = args.trial_source
    parent = args.parent
    strategy = args.strategy or ""

    if not os.path.exists(trial_source):
        print(f"Error: Trial source '{trial_source}' not found.", file=sys.stderr)
        sys.exit(1)

    state = _load_state(kernel_name)

    # Validate parent
    if parent is not None and parent not in state["trials"]:
        if state["next_id"] == 0:
            print(
                f"Warning: Ignoring --parent '{parent}' for first trial.",
                file=sys.stderr,
            )
            parent = None
        else:
            print(
                f"Error: Parent trial '{parent}' not found. Available: {list(state['trials'].keys())}",
                file=sys.stderr,
            )
            sys.exit(1)

    trial_id = f"t{state['next_id']}"
    state["next_id"] += 1

    # Copy files
    dest = os.path.join(_trial_dir(kernel_name), trial_id)
    try:
        if os.path.isdir(trial_source):
            shutil.copytree(trial_source, dest, dirs_exist_ok=True)
        else:
            os.makedirs(dest, exist_ok=True)
            shutil.copy2(trial_source, dest)
    except Exception as e:
        print(f"Warning: Copy issue: {e}", file=sys.stderr)

    state["trials"][trial_id] = {
        "parent": parent,
        "dir": trial_id,
        "strategy": strategy,
        "correctness": None,
        "speedup": None,
        "baseline_us": None,
        "kernel_us": None,
        "status": "saved",
    }
    _save_state(kernel_name, state)
    print(f"Saved trial {trial_id}: {strategy}")
    print(f"  Parent: {parent or 'root'}")
    print(f"  Dir: trials/{kernel_name}/{trial_id}/")


def cmd_result(args):
    """Record results for a trial."""
    kernel_name = args.kernel_name
    trial_id = args.trial_id

    state = _load_state(kernel_name)

    if trial_id not in state["trials"]:
        print(f"Error: Trial '{trial_id}' not found. Available: {list(state['trials'].keys())}", file=sys.stderr)
        sys.exit(1)

    trial = state["trials"][trial_id]

    if args.correctness:
        trial["correctness"] = args.correctness
    if args.speedup is not None:
        trial["speedup"] = args.speedup
    if args.baseline_us is not None:
        trial["baseline_us"] = args.baseline_us
    if args.kernel_us is not None:
        trial["kernel_us"] = args.kernel_us

    # Cache baseline_us at kernel level
    if args.baseline_us is not None and state.get("baseline_us") is None:
        state["baseline_us"] = [args.baseline_us]

    # Update status
    if trial["correctness"] == "fail":
        trial["status"] = "failed"
    elif trial["correctness"] == "pass" and trial["speedup"] is not None:
        trial["status"] = "completed"
    else:
        trial["status"] = "partial"

    # Update best trial
    best_speedup = -1.0
    best_id = None
    for tid, t in state["trials"].items():
        if t.get("correctness") == "pass" and t.get("speedup") is not None:
            if t["speedup"] > best_speedup:
                best_speedup = t["speedup"]
                best_id = tid
    state["best_trial"] = best_id

    _save_state(kernel_name, state)

    status_icon = {"completed": "+", "failed": "X", "partial": "~", "saved": "?"}
    icon = status_icon.get(trial["status"], "?")
    runtime_str = ""
    if trial.get("baseline_us") is not None and trial.get("kernel_us") is not None:
        runtime_str = f", baseline={trial['baseline_us']:.2f}us, kernel={trial['kernel_us']:.2f}us"
    print(
        f"[{icon}] {trial_id}: correctness={trial['correctness']}, speedup={trial['speedup']}{runtime_str}"
    )
    if state["best_trial"]:
        best = state["trials"][state["best_trial"]]
        best_runtime = ""
        if best.get("baseline_us") is not None and best.get("kernel_us") is not None:
            best_runtime = (
                f", baseline={best['baseline_us']:.2f}us, kernel={best['kernel_us']:.2f}us"
            )
        print(f"  Best trial: {state['best_trial']} ({best['speedup']}x{best_runtime})")


def cmd_status(args):
    """Show trial tree status as ASCII tree."""
    kernel_name = args.kernel_name
    state = _load_state(kernel_name)

    print(f"Trial tree: {state['kernel_name']}")
    print(f"  Baseline: {state['baseline_file']}")
    print(f"  Best: {state['best_trial'] or 'none'}")
    print(f"  Trials: {len(state['trials'])}")
    print()

    if not state["trials"]:
        print("  (no trials yet)")
        return

    # Build children map
    children = {}
    roots = []
    for tid, t in state["trials"].items():
        parent = t["parent"]
        if parent is None:
            roots.append(tid)
        else:
            children.setdefault(parent, []).append(tid)

    def sort_key(tid):
        return int(tid[1:])

    roots.sort(key=sort_key)
    for k in children:
        children[k].sort(key=sort_key)

    def print_node(tid, prefix="", is_last=True):
        trial = state["trials"][tid]
        connector = "└── " if is_last else "├── "

        is_best = tid == state["best_trial"]
        status_icon = {"completed": "+", "failed": "X", "partial": "~", "saved": "?"}
        icon = status_icon.get(trial["status"], "?")

        speedup_str = f"{trial['speedup']:.2f}x" if trial["speedup"] is not None else "---"
        runtime_str = ""
        if trial.get("baseline_us") is not None and trial.get("kernel_us") is not None:
            runtime_str = f" (bl={trial['baseline_us']:.0f}us, kr={trial['kernel_us']:.0f}us)"
        best_marker = " <<<< BEST" if is_best else ""
        strategy_short = trial["strategy"][:60] if trial["strategy"] else ""

        print(
            f"{prefix}{connector}[{icon}] {tid}: {speedup_str}{runtime_str} | {strategy_short}{best_marker}"
        )

        child_prefix = prefix + ("    " if is_last else "│   ")
        kids = children.get(tid, [])
        for i, child in enumerate(kids):
            print_node(child, child_prefix, i == len(kids) - 1)

    for i, root in enumerate(roots):
        print_node(root, "  ", i == len(roots) - 1)


def cmd_best(args):
    """Get the best trial info."""
    kernel_name = args.kernel_name
    state = _load_state(kernel_name)

    if state["best_trial"] is None:
        print("No correct trials yet.")
        sys.exit(1)

    best_id = state["best_trial"]
    best = state["trials"][best_id]
    best_dir = os.path.join(_trial_dir(kernel_name), best["dir"])

    print(f"best_trial: {best_id}")
    print(f"speedup: {best['speedup']}")
    if best.get("baseline_us") is not None:
        print(f"baseline_us: {best['baseline_us']}")
    if best.get("kernel_us") is not None:
        print(f"kernel_us: {best['kernel_us']}")
    print(f"strategy: {best['strategy']}")
    print(f"dir: {best_dir}")
    print(f"parent: {best['parent'] or 'root'}")


def cmd_baseline_us(args):
    """Print cached baseline time(s)."""
    kernel_name = args.kernel_name
    state = _load_state(kernel_name)

    baseline_us = state.get("baseline_us")
    if baseline_us is None:
        print(
            "No baseline_us cached yet. Run benchmark and record result for t0 first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(",".join(f"{v:.2f}" for v in baseline_us))


def cmd_finalize(args):
    """Copy the best correct trial to the output path."""
    kernel_name = args.kernel_name
    output_path = args.output_path

    state = _load_state(kernel_name)

    if state["best_trial"] is None:
        print("Error: No correct trials to finalize.", file=sys.stderr)
        sys.exit(1)

    best_id = state["best_trial"]
    best = state["trials"][best_id]
    src = os.path.join(_trial_dir(kernel_name), best["dir"])

    # Default bare names into output/
    if os.path.dirname(output_path) == "":
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, output_path)

    if os.path.isdir(src):
        shutil.copytree(src, output_path, dirs_exist_ok=True)
    else:
        shutil.copy2(src, output_path)

    runtime_str = ""
    if best.get("baseline_us") is not None and best.get("kernel_us") is not None:
        runtime_str = f", baseline={best['baseline_us']:.2f}us, kernel={best['kernel_us']:.2f}us"
    print(f"Finalized {best_id} ({best['speedup']}x{runtime_str}) -> {output_path}")
    print(f"  Strategy: {best['strategy']}")


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Trial Tree State Manager (CPU Kernels)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = subparsers.add_parser("init", help="Initialize trial tree")
    p_init.add_argument("kernel_name", help="Kernel identifier (e.g., rmsnorm)")
    p_init.add_argument("baseline_file", help="Path to PyTorch baseline file")

    # save
    p_save = subparsers.add_parser("save", help="Save a trial")
    p_save.add_argument("kernel_name", help="Kernel identifier")
    p_save.add_argument("trial_source", help="Path to the trial kernel dir or file")
    p_save.add_argument("--parent", default=None, help="Parent trial ID (e.g., t0)")
    p_save.add_argument("--strategy", default="", help="Description of optimization strategy")

    # result
    p_result = subparsers.add_parser("result", help="Record trial results")
    p_result.add_argument("kernel_name", help="Kernel identifier")
    p_result.add_argument("trial_id", help="Trial ID (e.g., t0)")
    p_result.add_argument("--correctness", choices=["pass", "fail"], help="Correctness result")
    p_result.add_argument("--speedup", type=float, help="Speedup over baseline")
    p_result.add_argument("--baseline_us", type=float, help="Baseline runtime in microseconds")
    p_result.add_argument("--kernel_us", type=float, help="Kernel runtime in microseconds")

    # status
    p_status = subparsers.add_parser("status", help="Show trial tree status")
    p_status.add_argument("kernel_name", help="Kernel identifier")

    # best
    p_best = subparsers.add_parser("best", help="Get best trial info")
    p_best.add_argument("kernel_name", help="Kernel identifier")

    # baseline-us
    p_baseline_us = subparsers.add_parser("baseline-us", help="Print cached baseline time(s)")
    p_baseline_us.add_argument("kernel_name", help="Kernel identifier")

    # finalize
    p_finalize = subparsers.add_parser("finalize", help="Copy best trial to output")
    p_finalize.add_argument("kernel_name", help="Kernel identifier")
    p_finalize.add_argument("output_path", help="Output path (bare name defaults to output/)")

    args = parser.parse_args()

    commands = {
        "init": cmd_init,
        "save": cmd_save,
        "result": cmd_result,
        "status": cmd_status,
        "best": cmd_best,
        "baseline-us": cmd_baseline_us,
        "finalize": cmd_finalize,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
