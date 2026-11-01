from __future__ import annotations

import logging
import os
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .store import Database


logger = logging.getLogger(__name__)

FORCED_CANCEL_STAGES = frozenset({"run_asr", "align_segments"})
DEFAULT_TERMINATION_GRACE_SECONDS = 3.0
SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)


@dataclass(frozen=True)
class WorkerIdentity:
    pid: int
    process_start_id: str
    command_line: str
    process_group_id: int


@dataclass(frozen=True)
class WorkerShutdownResult:
    requested: bool
    reason: str
    pid: int | None = None


def _read_proc_stat(pid: int) -> tuple[int, str]:
    stat_path = Path(f"/proc/{pid}/stat")
    content = stat_path.read_text(encoding="utf-8")
    command_end = content.rfind(")")
    if command_end < 0:
        raise RuntimeError(f"invalid /proc/{pid}/stat")
    fields = content[command_end + 2 :].split()
    # fields[0] is stat field 3 (state); starttime is field 22, pgrp is field 5.
    if len(fields) < 20:
        raise RuntimeError(f"incomplete /proc/{pid}/stat")
    return int(fields[2]), fields[19]


def read_process_identity(pid: int) -> WorkerIdentity:
    """Read the Linux process identity used to reject stale/reused PIDs."""
    process_group_id, process_start_id = _read_proc_stat(pid)
    raw_command_line = Path(f"/proc/{pid}/cmdline").read_bytes()
    command_line = " ".join(
        part.decode("utf-8", errors="replace")
        for part in raw_command_line.split(b"\0")
        if part
    ).strip()
    if not command_line:
        raise RuntimeError(f"process {pid} has no command line")
    return WorkerIdentity(
        pid=pid,
        process_start_id=process_start_id,
        command_line=command_line,
        process_group_id=process_group_id,
    )


def isolate_worker_process_group() -> None:
    """Ensure the Worker PID is also its process-group ID when supported."""
    if not hasattr(os, "setsid"):
        return
    try:
        os.setsid()
    except OSError:
        if os.getpgrp() != os.getpid():
            raise


def _identity_matches(expected: dict[str, Any], actual: WorkerIdentity) -> bool:
    return (
        int(expected["pid"]) == actual.pid
        and str(expected["process_start_id"]) == actual.process_start_id
        and str(expected["command_line"]).strip() == actual.command_line.strip()
    )


def _signal_isolated_worker_group(identity: WorkerIdentity, sig: int) -> None:
    if identity.process_group_id != identity.pid:
        raise RuntimeError("worker process group is not isolated")
    if not hasattr(os, "killpg"):
        raise RuntimeError("process-group signalling is unavailable")
    os.killpg(identity.process_group_id, sig)


def _force_kill_after_grace(identity: WorkerIdentity, grace_seconds: float) -> None:
    time.sleep(max(grace_seconds, 0))
    try:
        current = read_process_identity(identity.pid)
    except (FileNotFoundError, ProcessLookupError, RuntimeError):
        return
    expected = {
        "pid": identity.pid,
        "process_start_id": identity.process_start_id,
        "command_line": identity.command_line,
    }
    if not _identity_matches(expected, current):
        logger.warning("Worker PID %d 已被复用，跳过 SIGKILL", identity.pid)
        return
    try:
        _signal_isolated_worker_group(current, SIGKILL)
    except (ProcessLookupError, PermissionError, RuntimeError):
        logger.exception("强制终止 Worker 失败 (pid=%d)", identity.pid)


def request_forced_worker_shutdown(
    database: Database,
    task_id: int,
    *,
    grace_seconds: float = DEFAULT_TERMINATION_GRACE_SECONDS,
    schedule_force_kill: bool = True,
) -> WorkerShutdownResult:
    """Terminate the Worker only if it still owns the target forced-cancel stage."""
    worker = database.get_worker_runtime()
    if worker is None:
        return WorkerShutdownResult(False, "worker_not_registered")
    if worker.get("current_task_id") != task_id:
        return WorkerShutdownResult(False, "worker_not_processing_task", int(worker["pid"]))
    if str(worker.get("current_stage") or "") not in FORCED_CANCEL_STAGES:
        return WorkerShutdownResult(False, "stage_not_force_cancellable", int(worker["pid"]))

    try:
        identity = read_process_identity(int(worker["pid"]))
    except (FileNotFoundError, ProcessLookupError, RuntimeError):
        return WorkerShutdownResult(False, "worker_process_stale", int(worker["pid"]))
    if not _identity_matches(worker, identity):
        return WorkerShutdownResult(False, "worker_identity_mismatch", int(worker["pid"]))
    if identity.process_group_id != identity.pid:
        return WorkerShutdownResult(False, "worker_process_group_not_isolated", identity.pid)

    try:
        _signal_isolated_worker_group(identity, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, RuntimeError):
        logger.exception("请求终止 Worker 失败 (pid=%d)", identity.pid)
        return WorkerShutdownResult(False, "worker_signal_failed", identity.pid)

    if schedule_force_kill:
        threading.Thread(
            target=_force_kill_after_grace,
            args=(identity, grace_seconds),
            name=f"worker-force-kill-{identity.pid}",
            daemon=True,
        ).start()
    return WorkerShutdownResult(True, "termination_requested", identity.pid)


def _descendant_pids(pid: int) -> list[int]:
    children_path = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        direct_children = [int(value) for value in children_path.read_text(encoding="utf-8").split()]
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return []
    descendants: list[int] = []
    for child in direct_children:
        descendants.append(child)
        descendants.extend(_descendant_pids(child))
    return descendants


def terminate_descendant_processes(pid: int | None = None, grace_seconds: float = 1.0) -> None:
    """Terminate Worker children, primarily FFmpeg, when the Worker is signalled."""
    parent_pid = pid or os.getpid()
    descendants = _descendant_pids(parent_pid)
    if not descendants:
        return
    for child_pid in descendants:
        try:
            os.kill(child_pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    if grace_seconds <= 0:
        return
    time.sleep(grace_seconds)
    remaining_descendants = set(_descendant_pids(parent_pid))
    for child_pid in descendants:
        if child_pid not in remaining_descendants:
            continue
        try:
            os.kill(child_pid, SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def install_worker_signal_handlers() -> None:
    def handle_shutdown(signum: int, _frame: Any) -> None:
        logger.warning("Worker 收到信号 %s，开始清理子进程", signum)
        terminate_descendant_processes()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)
