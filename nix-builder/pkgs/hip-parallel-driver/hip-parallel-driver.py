#! @python3@ -I
"""Run a clang compilation with independent driver jobs in parallel.

Usage: hip-parallel-driver JOBS CLANG ARGS...

When clang compiles HIP code for multiple offload architectures, the driver
runs the per-architecture device compilation and linking jobs one after
another. This script asks clang for its job plan (`-###`) and then runs the
jobs itself, running independent jobs (e.g. the per-architecture ones)
concurrently. A job is only started after all jobs that produce one of its
inputs have finished.
"""

import os
import shlex
import subprocess
import sys
import tempfile


def outputs(job):
    outs = []
    for i, arg in enumerate(job):
        if arg == "-o" and i + 1 < len(job):
            outs.append(job[i + 1])
        elif arg.startswith("-output="):
            outs.append(arg[len("-output=") :])
    return outs


def main():
    max_jobs = int(sys.argv[1])
    clang = sys.argv[2]
    args = sys.argv[3:]

    plan = subprocess.run(
        [clang, *args, "-###"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if plan.returncode != 0:
        sys.stderr.write(plan.stderr)
        return plan.returncode

    jobs = [
        shlex.split(line) for line in plan.stderr.splitlines() if line.startswith(' "')
    ]

    # Nothing to parallelize or intermediate files are requested: let clang
    # do the work.
    if len(jobs) < 2 or any(a.startswith("-save-temps") for a in args):
        os.execv(clang, [clang, *args])

    # Forward diagnostics that the driver produced while planning.
    for line in plan.stderr.splitlines():
        if ": warning:" in line or ": error:" in line or ": note:" in line:
            sys.stderr.write(line + "\n")

    job_outs = [outputs(j) for j in jobs]
    deps = [
        {k for k in range(i) if any(o in a for a in job for o in job_outs[k])}
        for i, job in enumerate(jobs)
    ]

    pending = list(range(len(jobs)))
    done = set()
    running = {}
    status = 0

    def finish(pid, code):
        nonlocal status
        i, proc, err = running.pop(pid)
        proc.returncode = code
        err.seek(0)
        sys.stderr.write(err.read().decode(errors="replace"))
        sys.stderr.flush()
        err.close()
        if code != 0 and status == 0:
            status = code
            for _, other, _ in running.values():
                other.terminate()
        done.add(i)

    try:
        while pending or running:
            if status == 0:
                for i in list(pending):
                    if len(running) >= max_jobs:
                        break
                    if deps[i] <= done:
                        pending.remove(i)
                        err = tempfile.TemporaryFile()
                        proc = subprocess.Popen(jobs[i], stderr=err)
                        running[proc.pid] = (i, proc, err)
            elif not running:
                break
            if not running:
                break
            pid, st = os.wait()
            if pid in running:
                finish(pid, os.waitstatus_to_exitcode(st))
    finally:
        # Stop remaining jobs on failure/interruption.
        for pid, (_, proc, _) in list(running.items()):
            proc.terminate()
        for pid in list(running):
            try:
                _, st = os.waitpid(pid, 0)
                finish(pid, os.waitstatus_to_exitcode(st))
            except ChildProcessError:
                running.pop(pid)

        # Remove intermediate files (outputs consumed by another job) that the
        # driver would have cleaned up. Final outputs are never consumed by a
        # job, so they are kept, wherever they are located.
        consumed = {k for d in deps for k in d}
        for k in consumed:
            for out in job_outs[k]:
                try:
                    os.unlink(out)
                except OSError:
                    pass

    return status if status >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
