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
import signal
import subprocess
import sys
import tempfile
from typing import IO


def outputs(job: list[str]) -> list[str]:
    outs: list[str] = []
    for i, arg in enumerate(job):
        if arg == "-o" and i + 1 < len(job):
            outs.append(job[i + 1])
        elif arg.startswith("-output="):
            outs.append(arg[len("-output=") :])
    return outs


def main() -> int:
    # At least one job, otherwise nothing would ever be started.
    max_jobs = max(1, int(sys.argv[1]))
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

    # deps[i] holds the indices of the earlier jobs that job i depends on.
    deps = [
        {
            earlier
            for earlier in range(job_idx)
            if any(
                output in arg  # substring: also matches e.g. `-input=<path>`
                for arg in job_args  # every argument of this job
                for output in job_outs[earlier]  # every file the earlier job writes
            )
        }
        for job_idx, job_args in enumerate(jobs)
    ]

    pending = list(range(len(jobs)))
    done: set[int] = set()
    # pid -> (job index, file that captures the stderr of the job)
    running: dict[int, tuple[int, IO[bytes]]] = {}
    status = 0

    def stop_running() -> None:
        for pid in running:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def finish(pid: int, code: int) -> None:
        nonlocal status
        i, err = running.pop(pid)
        err.seek(0)
        sys.stderr.write(err.read().decode(errors="replace"))
        sys.stderr.flush()
        err.close()
        if code != 0 and status == 0:
            status = code
            stop_running()
        done.add(i)

    try:
        while pending or running:
            if status == 0:
                for i in list(pending):
                    # Do not spawn more jobs than requested.
                    if len(running) >= max_jobs:
                        break
                    # Spawn this pending job if its dependencies have finished.
                    if deps[i] <= done:
                        pending.remove(i)
                        err = tempfile.TemporaryFile()
                        try:
                            pid = os.posix_spawn(
                                jobs[i][0],
                                jobs[i],
                                os.environ,
                                file_actions=[(os.POSIX_SPAWN_DUP2, err.fileno(), 2)],
                            )
                        except OSError as exc:
                            err.close()
                            sys.stderr.write(f"{jobs[i][0]}: {exc.strerror}\n")
                            status = 127
                            stop_running()
                            break
                        running[pid] = (i, err)
            if not running:
                break
            pid, st = os.wait()
            finish(pid, os.waitstatus_to_exitcode(st))
    finally:
        # Stop and collect remaining jobs on failure/interruption. This
        # should not be needed in normal cases, since when a job fails,
        # finish() already calls `stop_running()`. However, this is an
        # extra safeguard for e.g. when the loop raises and therefore
        # finish() does not get called.
        stop_running()
        while running:
            pid, st = os.wait()
            finish(pid, os.waitstatus_to_exitcode(st))

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
