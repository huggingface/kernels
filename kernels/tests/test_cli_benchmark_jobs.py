import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from huggingface_hub.errors import LocalTokenNotFoundError

from kernels._internals.cli import benchmark_jobs, main


def run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["kernels", "benchmark", *args])
    main()


@pytest.fixture
def jobs(monkeypatch):
    api = Mock()
    api.run_job.side_effect = lambda *, flavor, **kwargs: SimpleNamespace(url=f"https://example.org/jobs/{flavor}")
    monkeypatch.setattr(benchmark_jobs, "_get_hf_api", lambda: api)
    return api.run_job


@pytest.mark.parametrize(
    ("target", "worker_args"),
    [
        (["org/kernel@v2"], ["org/kernel@v2"]),
        (["org/kernel", "--version", "0"], ["org/kernel", "--version", "0"]),
        (["org/kernel@feature/test"], ["org/kernel@feature/test"]),
        (["org/kernel", "--branch", "feature/test"], ["org/kernel", "--branch=feature/test"]),
    ],
)
def test_submit_benchmark_jobs(monkeypatch, jobs, target, worker_args, capsys):
    run_cli(
        monkeypatch,
        *target,
        "--flavor",
        "l4x1",
        "h200",
        "l4x1",
        "--iterations",
        "20",
        "--warmup",
        "0",
        "--json",
        "--show-stats",
    )

    assert [call.kwargs["flavor"] for call in jobs.call_args_list] == ["l4x1", "h200"]
    for call in jobs.call_args_list:
        command = call.kwargs["command"]
        assert command[command.index(target[0]) :] == [
            *worker_args,
            "--iterations",
            "20",
            "--warmup",
            "0",
            "--json",
            "--show-stats",
        ]
        assert "secrets" not in call.kwargs
    assert capsys.readouterr().out.splitlines() == [
        "l4x1: https://example.org/jobs/l4x1",
        "h200: https://example.org/jobs/h200",
    ]


@pytest.mark.parametrize(
    "args",
    [
        ["."],
        ["org/kernel"],
        ["org/kernel@2"],
        ["org/kernel@v2", "--output", "result.jsonl"],
        ["org/kernel@v2", "--baseline", "base.jsonl"],
        ["org/kernel@v2", "--iterations", "0"],
    ],
)
def test_invalid_submission_creates_no_job(monkeypatch, jobs, args):
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, *args, "--flavor", "h200")
    jobs.assert_not_called()


def test_submission_failure_preserves_earlier_job_url(monkeypatch, jobs, capsys):
    jobs.side_effect = [
        SimpleNamespace(url="https://example.org/jobs/first"),
        LocalTokenNotFoundError("login required"),
    ]
    with pytest.raises(SystemExit, match="h200: login required"):
        run_cli(monkeypatch, "org/kernel@main", "--flavor", "l4x1", "h200")
    assert "https://example.org/jobs/first" in capsys.readouterr().out
