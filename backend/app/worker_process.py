from __future__ import annotations

import logging
import os
import sys
import threading

from .logging_utils import setup_logging
from .main import resolve_db_path
from .runtime import WorkerService
from .store import Database
from .worker_control import (
    WorkerIdentity,
    install_worker_signal_handlers,
    isolate_worker_process_group,
    read_process_identity,
)


logger = logging.getLogger(__name__)
WORKER_HEARTBEAT_INTERVAL_SECONDS = 5.0


def _start_heartbeat(database: Database, identity: WorkerIdentity, stop_event: threading.Event) -> threading.Thread:
    def heartbeat_loop() -> None:
        heartbeat_database = Database(str(database.db_path))
        while not stop_event.wait(WORKER_HEARTBEAT_INTERVAL_SECONDS):
            try:
                heartbeat_database.heartbeat_worker(identity.pid, identity.process_start_id)
            except Exception:
                logger.exception("Worker 心跳写入失败")

    thread = threading.Thread(target=heartbeat_loop, name="worker-heartbeat", daemon=True)
    thread.start()
    return thread


def main() -> None:
    database = Database(resolve_db_path(), persistent=True)
    database.initialize()
    setup_logging(database)
    identity: WorkerIdentity | None = None
    if sys.platform.startswith("linux"):
        isolate_worker_process_group()
        identity = read_process_identity(os.getpid())
        database.register_worker(
            identity.pid,
            identity.process_start_id,
            identity.command_line,
            identity.process_group_id,
        )
    else:
        logger.warning("当前平台不支持 Worker 进程组强制取消，仅启用协作取消")

    stop_event = threading.Event()
    if identity is not None:
        _start_heartbeat(database, identity, stop_event)
        install_worker_signal_handlers()
    database.recover_orphaned_tasks()
    database.clear_restart_required()
    try:
        WorkerService(database, worker_identity=identity).run_forever()
    finally:
        stop_event.set()
        database.close()


if __name__ == "__main__":
    main()
