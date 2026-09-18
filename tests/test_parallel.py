"""Tests for the parallel dispatch layer: thread budgeting, run_bash_command,
and failure propagation through the real dispatchers.

Unlike the orchestration tests (which monkeypatch the dispatchers away), the
tests here execute parallel_bash_commands / serial_bash_commands for real, so
a ProcessPoolExecutor worker is actually spawned. On Python >= 3.14 the
default start method re-imports this module in the worker, which is exactly
the regression class the explicit ``dryrun`` argument guards against.
"""

import pytest

from undistortme import pipeline as cp


@pytest.fixture(autouse=True)
def _reset_failed_commands(monkeypatch):
    """Isolate the module-level failure accumulator per test."""
    monkeypatch.setattr(cp, "failed_commands", [])


# ===========================================================================
# topup_threads / default_jobs
# ===========================================================================


class TestTopupThreads:
    """nthr = clamp(round(jobs * oversub / min(jobs, batch)), 1, jobs)."""

    @pytest.mark.parametrize(
        "jobs,oversub,batch,expected",
        [
            (8, 4.0, 1, 8),  # single command: full jobs (clamped)
            (8, 4.0, 8, 4),  # saturated batch: jobs*4/8
            (8, 4.0, 32, 4),  # batch beyond jobs: concurrency capped at jobs
            (8, 1.0, 8, 1),  # strict budget: one thread each
            (8, 2.0, 8, 2),
            (8, 0.01, 8, 1),  # floor at 1
            (2, 8.0, 1, 2),  # never exceeds jobs
        ],
    )
    def test_budget_formula(self, monkeypatch, jobs, oversub, batch, expected):
        monkeypatch.setattr(
            cp, "check_dict", {"jobs": jobs, "oversubscribe": oversub}, raising=False
        )
        assert cp.topup_threads(batch) == expected

    def test_fallback_to_defaults(self, monkeypatch):
        """Missing check_dict keys fall back to default_jobs / DEFAULT_OVERSUBSCRIBE."""
        monkeypatch.setattr(cp, "check_dict", {}, raising=False)
        monkeypatch.setattr(cp, "default_jobs", lambda: 8)
        expected = int(max(1, min(8, round(8 * cp.DEFAULT_OVERSUBSCRIBE / 8))))
        assert cp.topup_threads(8) == expected

    def test_default_jobs_is_positive_int(self):
        jobs = cp.default_jobs()
        assert isinstance(jobs, int) and jobs >= 1


# ===========================================================================
# run_bash_command
# ===========================================================================


class TestRunBashCommand:
    def test_dryrun_returns_zero_without_running(self, tmp_path):
        marker = tmp_path / "ran"
        cmd, rc, err = cp.run_bash_command(f"touch {marker}", dryrun=True)
        assert rc == 0 and err == ""
        assert not marker.exists()

    def test_success(self):
        cmd, rc, err = cp.run_bash_command("true", dryrun=False)
        assert (cmd, rc, err) == ("true", 0, "")

    def test_failure_returns_code_and_output_tail(self):
        cmd, rc, err = cp.run_bash_command("echo boom; exit 7", dryrun=False)
        assert rc == 7
        assert "boom" in err


# ===========================================================================
# Dispatchers (REAL execution) + failure accumulation
# ===========================================================================


class TestDispatcherFailurePropagation:
    def _gates(self, monkeypatch, **over):
        d = {"dryrun": False, "jobs": 2}
        d.update(over)
        monkeypatch.setattr(cp, "check_dict", d, raising=False)

    def test_serial_collects_failures(self, monkeypatch):
        self._gates(monkeypatch)
        cp.serial_bash_commands(["true", "false", "true"], "serial test")
        assert len(cp.failed_commands) == 1
        assert cp.failed_commands[0] == ("false", 1)

    def test_parallel_collects_failures_with_real_workers(self, monkeypatch):
        """Spawns a real ProcessPoolExecutor (py>=3.14 worker-safety check)."""
        self._gates(monkeypatch)
        cp.parallel_bash_commands(["true", "false", "exit 3"], "parallel test")
        codes = sorted(rc for _cmd, rc in cp.failed_commands)
        assert codes == [1, 3]

    def test_none_and_empty_batches_are_noops(self, monkeypatch):
        self._gates(monkeypatch)
        cp.parallel_bash_commands(None, "none batch")
        cp.parallel_bash_commands([None, None], "all-None batch")
        cp.serial_bash_commands(None, "none batch")
        assert cp.failed_commands == []

    def test_dryrun_never_fails(self, monkeypatch):
        self._gates(monkeypatch, dryrun=True)
        cp.serial_bash_commands(["false"], "dryrun serial")
        cp.parallel_bash_commands(["false"], "dryrun parallel")
        assert cp.failed_commands == []
