from __future__ import annotations

import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from app.main import create_app
from app.runtime import WorkerService
from app.store import Database
from app.worker_process import main as worker_main
from app.worker_control import (
    WorkerIdentity,
    WorkerShutdownResult,
    _read_proc_stat,
    request_forced_worker_shutdown,
    terminate_descendant_processes,
)


WORKER_PID = 43210
WORKER_START_ID = "987654"
WORKER_COMMAND = "python3 -m app.worker_process"


class ForcedCancellationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base = Path(self.temp_dir.name)
        self.data_dir = self.base / "data"
        self.output_dir = self.base / "output"
        self.config_dir = self.base / "config"
        self.models_dir = self.base / "models"
        self.frontend_dist = self.base / "frontend"
        for path in (self.data_dir, self.output_dir, self.config_dir, self.models_dir, self.frontend_dist):
            path.mkdir(parents=True, exist_ok=True)
        (self.frontend_dist / "index.html").write_text("<!doctype html><title>test</title>", encoding="utf-8")

        self.db_path = self.config_dir / "forced-cancel.db"
        self.database = Database(str(self.db_path))
        self.database.initialize()
        self.database.update_config(
            {
                "processing": {
                    "work_dir": str(self.config_dir / "work"),
                    "max_retries": 0,
                }
            }
        )
        self.env_patch = patch.dict(
            os.environ,
            {
                "SUBPIPELINE_DB_PATH": str(self.db_path),
                "SUBPIPELINE_FRONTEND_DIST": str(self.frontend_dist),
                "SUBPIPELINE_MODELS_DIR": str(self.models_dir),
                "SUBPIPELINE_BROWSE_ROOTS": ",".join(
                    str(path) for path in (self.data_dir, self.output_dir, self.config_dir)
                ),
                "SUBPIPELINE_OUTPUT_DIR": str(self.output_dir),
            },
            clear=False,
        )
        self.env_patch.start()
        self.task_sequence = 0

    def tearDown(self) -> None:
        self.database.close()
        self.env_patch.stop()
        self.temp_dir.cleanup()

    def _create_task(self) -> dict:
        self.task_sequence += 1
        video_path = self.data_dir / f"forced-cancel-{self.task_sequence}.mp4"
        video_path.write_bytes(b"video")
        observed = self.database.observe_file(
            str(video_path),
            int(video_path.stat().st_size),
            float(video_path.stat().st_mtime),
        )
        return self.database.create_task(
            observed["file_id"],
            observed["path"],
            observed["size_bytes"],
            observed["mtime"],
        )

    def _claim_asr_task(self) -> dict:
        task = self._create_task()
        claimed = self.database.claim_next_pending_task()
        self.assertIsNotNone(claimed)
        assert claimed is not None
        self.database.update_task_stage(claimed["id"], "run_asr", 35)
        return self.database.get_task(claimed["id"])

    def _register_worker(self, task: dict | None = None, stage: str | None = None) -> None:
        self.database.register_worker(
            WORKER_PID,
            WORKER_START_ID,
            WORKER_COMMAND,
            WORKER_PID,
        )
        if task is not None:
            self.database.update_worker_task(
                WORKER_PID,
                WORKER_START_ID,
                int(task["id"]),
                stage,
            )

    def test_proc_stat_reads_process_group_and_start_id(self) -> None:
        fields = ["S", "0", str(WORKER_PID)] + ["0"] * 16 + [WORKER_START_ID]
        stat = f"123 (python worker) {' '.join(fields)}"
        with patch("app.worker_control.Path") as path_type:
            path_type.return_value.read_text.return_value = stat
            process_group_id, process_start_id = _read_proc_stat(123)

        self.assertEqual(process_group_id, WORKER_PID)
        self.assertEqual(process_start_id, WORKER_START_ID)

    def test_worker_registration_heartbeat_and_current_stage_are_persisted(self) -> None:
        task = self._claim_asr_task()
        self._register_worker(task, "run_asr")

        worker = self.database.get_worker_runtime()
        self.assertIsNotNone(worker)
        assert worker is not None
        self.assertEqual(worker["pid"], WORKER_PID)
        self.assertEqual(worker["process_start_id"], WORKER_START_ID)
        self.assertEqual(worker["command_line"], WORKER_COMMAND)
        self.assertEqual(worker["current_task_id"], task["id"])
        self.assertEqual(worker["current_stage"], "run_asr")

        self.database.update_task_stage(task["id"], "align_segments", 45)
        synced = self.database.get_worker_runtime()
        self.assertIsNotNone(synced)
        assert synced is not None
        self.assertEqual(synced["current_stage"], "align_segments")

        with self.database.connect() as connection:
            connection.execute(
                "UPDATE worker_runtime SET heartbeat_at = '2000-01-01T00:00:00+00:00' WHERE id = 1"
            )
        self.assertTrue(self.database.heartbeat_worker(WORKER_PID, WORKER_START_ID))
        self.assertFalse(self.database.heartbeat_worker(WORKER_PID, "different-start"))
        refreshed = self.database.get_worker_runtime()
        self.assertIsNotNone(refreshed)
        assert refreshed is not None
        self.assertNotEqual(refreshed["heartbeat_at"], "2000-01-01T00:00:00+00:00")

        identity = WorkerIdentity(WORKER_PID, WORKER_START_ID, WORKER_COMMAND, WORKER_PID)
        WorkerService(self.database, worker_identity=identity)._set_worker_task(None, None)
        cleared = self.database.get_worker_runtime()
        self.assertIsNotNone(cleared)
        assert cleared is not None
        self.assertIsNone(cleared["current_task_id"])
        self.assertIsNone(cleared["current_stage"])

    def test_stale_start_id_or_command_line_does_not_send_signal(self) -> None:
        task = self._claim_asr_task()
        self._register_worker(task, "run_asr")

        cases = [
            WorkerIdentity(WORKER_PID, "different-start", WORKER_COMMAND, WORKER_PID),
            WorkerIdentity(WORKER_PID, WORKER_START_ID, "python3 -m app.other_process", WORKER_PID),
        ]
        for identity in cases:
            with self.subTest(identity=identity), patch(
                "app.worker_control.read_process_identity", return_value=identity
            ), patch("app.worker_control.os.killpg", create=True) as killpg:
                result = request_forced_worker_shutdown(
                    self.database,
                    task["id"],
                    schedule_force_kill=False,
                )

            self.assertFalse(result.requested)
            self.assertEqual(result.reason, "worker_identity_mismatch")
            killpg.assert_not_called()

    def test_stale_pid_does_not_send_signal(self) -> None:
        task = self._claim_asr_task()
        self._register_worker(task, "run_asr")

        with patch(
            "app.worker_control.read_process_identity", side_effect=FileNotFoundError()
        ), patch("app.worker_control.os.killpg", create=True) as killpg:
            result = request_forced_worker_shutdown(
                self.database,
                task["id"],
                schedule_force_kill=False,
            )

        self.assertFalse(result.requested)
        self.assertEqual(result.reason, "worker_process_stale")
        killpg.assert_not_called()

    def test_force_cancel_sends_sigterm_to_validated_worker_group(self) -> None:
        task = self._claim_asr_task()
        self._register_worker(task, "align_segments")
        identity = WorkerIdentity(WORKER_PID, WORKER_START_ID, WORKER_COMMAND, WORKER_PID)

        with patch("app.worker_control.read_process_identity", return_value=identity), patch(
            "app.worker_control.os.killpg", create=True
        ) as killpg:
            result = request_forced_worker_shutdown(
                self.database,
                task["id"],
                schedule_force_kill=False,
            )

        self.assertTrue(result.requested)
        self.assertEqual(result.reason, "termination_requested")
        killpg.assert_called_once_with(WORKER_PID, signal.SIGTERM)

    def test_asr_cancel_requests_worker_termination_once_and_returns_cancelling(self) -> None:
        task = self._claim_asr_task()
        self._register_worker(task, "run_asr")
        shutdown = WorkerShutdownResult(True, "termination_requested", WORKER_PID)

        with patch(
            "app.main.request_forced_worker_shutdown", return_value=shutdown
        ) as terminate, TestClient(create_app()) as client:
            first = client.post(f"/api/tasks/{task['id']}/cancel")
            second = client.post(f"/api/tasks/{task['id']}/cancel")

        self.assertEqual(first.status_code, 202)
        self.assertEqual(first.json()["outcome"], "cancelling")
        self.assertEqual(first.json()["status"], "processing")
        self.assertTrue(first.json()["cancel_requested"])
        self.assertEqual(first.json(), second.json())
        terminate.assert_called_once()
        call_args = terminate.call_args.args
        self.assertIsInstance(call_args[0], Database)
        self.assertEqual(call_args[1], task["id"])

    def test_recovery_cancels_requested_tasks_before_failing_other_orphans(self) -> None:
        cancelled_task = self._claim_asr_task()
        failed_task = self._claim_asr_task()
        self.database.request_cancel(cancelled_task["id"])

        recovered_count = self.database.recover_orphaned_tasks()

        self.assertEqual(recovered_count, 2)
        cancelled = self.database.get_task(cancelled_task["id"])
        failed = self.database.get_task(failed_task["id"])
        self.assertIsNotNone(cancelled)
        self.assertIsNotNone(failed)
        assert cancelled is not None and failed is not None
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertIn("Worker 已终止", cancelled["error_message"])
        self.assertEqual(failed["status"], "failed")
        self.assertIn("系统重启", failed["error_message"])

    def test_signal_cleanup_terminates_worker_children_without_real_signal(self) -> None:
        with patch("app.worker_control._descendant_pids", return_value=[101, 102]), patch(
            "app.worker_control.time.sleep"
        ), patch("app.worker_control.os.kill") as kill:
            terminate_descendant_processes(pid=100, grace_seconds=0.1)

        self.assertEqual(
            kill.call_args_list,
            [
                call(101, signal.SIGTERM),
                call(102, signal.SIGTERM),
                call(101, getattr(signal, "SIGKILL", signal.SIGTERM)),
                call(102, getattr(signal, "SIGKILL", signal.SIGTERM)),
            ],
        )

    def test_supervisor_stops_worker_process_group(self) -> None:
        config = (ROOT.parent / "container" / "supervisord.conf").read_text(encoding="utf-8")
        worker_section = config.split("[program:worker]", 1)[1]
        self.assertIn("stopasgroup=true", worker_section)
        self.assertIn("killasgroup=true", worker_section)
        self.assertIn("stopwaitsecs=10", worker_section)

    def test_worker_startup_registers_before_recovery(self) -> None:
        identity = WorkerIdentity(WORKER_PID, WORKER_START_ID, WORKER_COMMAND, WORKER_PID)
        database = MagicMock()
        database.db_path = self.db_path
        calls: list[str] = []
        database.register_worker.side_effect = lambda *_args: calls.append("register") or {}
        database.recover_orphaned_tasks.side_effect = lambda: calls.append("recover") or 0
        database.clear_restart_required.side_effect = lambda: calls.append("clear_restart")

        with patch("app.worker_process.Database", return_value=database), patch(
            "app.worker_process.setup_logging"
        ), patch("app.worker_process.sys.platform", "linux"), patch(
            "app.worker_process.isolate_worker_process_group"
        ), patch("app.worker_process.read_process_identity", return_value=identity), patch(
            "app.worker_process._start_heartbeat"
        ), patch("app.worker_process.install_worker_signal_handlers"), patch(
            "app.worker_process.WorkerService"
        ) as worker_service:
            worker_main()

        self.assertEqual(calls, ["register", "recover", "clear_restart"])
        database.register_worker.assert_called_once_with(
            WORKER_PID,
            WORKER_START_ID,
            WORKER_COMMAND,
            WORKER_PID,
        )
        worker_service.assert_called_once_with(database, worker_identity=identity)
        worker_service.return_value.run_forever.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
