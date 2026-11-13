from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from app.main import create_app
from app.store import Database


class TaskCancellationTests(unittest.TestCase):
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

        self.db_path = self.config_dir / "cancellation.db"
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
        video_path = self.data_dir / f"cancel-{self.task_sequence}.mp4"
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

    def _claim_processing_task(self) -> dict:
        self._create_task()
        task = self.database.claim_next_pending_task()
        self.assertIsNotNone(task)
        self.assertEqual(task["status"], "processing")
        return task

    def test_pending_cancel_is_immediate_and_cannot_be_claimed(self) -> None:
        task = self._create_task()

        with TestClient(create_app()) as client:
            response = client.post(f"/api/tasks/{task['id']}/cancel")
            detail = client.get(f"/api/tasks/{task['id']}")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["outcome"], "cancelled")
        self.assertEqual(payload["status"], "cancelled")
        self.assertTrue(payload["message"])
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["status"], "cancelled")
        stored = self.database.get_task(task["id"])
        self.assertIsNotNone(stored)
        self.assertEqual(stored["status"], "cancelled")
        self.assertIsNone(self.database.claim_next_pending_task())

    def test_processing_cancel_requests_stop_without_changing_status(self) -> None:
        task = self._claim_processing_task()

        with TestClient(create_app()) as client:
            response = client.post(f"/api/tasks/{task['id']}/cancel")
            detail = client.get(f"/api/tasks/{task['id']}")

        self.assertEqual(response.status_code, 202)
        payload = response.json()
        self.assertEqual(payload["outcome"], "cancelling")
        self.assertEqual(payload["status"], "processing")
        self.assertTrue(payload["cancel_requested"])
        self.assertTrue(payload["message"])
        self.assertEqual(detail.json()["status"], "processing")
        stored = self.database.get_task(task["id"])
        self.assertIsNotNone(stored)
        self.assertEqual(stored["status"], "processing")
        self.assertTrue(stored["cancel_requested"])

    def test_repeated_cancel_is_idempotent(self) -> None:
        processing_task = self._claim_processing_task()
        pending_task = self._create_task()

        with TestClient(create_app()) as client:
            pending_first = client.post(f"/api/tasks/{pending_task['id']}/cancel")
            pending_second = client.post(f"/api/tasks/{pending_task['id']}/cancel")
            processing_first = client.post(f"/api/tasks/{processing_task['id']}/cancel")
            processing_after_first = self.database.get_task(processing_task["id"])
            processing_second = client.post(f"/api/tasks/{processing_task['id']}/cancel")
            processing_after_second = self.database.get_task(processing_task["id"])

        self.assertEqual(pending_first.status_code, 200)
        self.assertEqual(pending_second.status_code, 200)
        self.assertEqual(pending_first.json(), pending_second.json())
        self.assertEqual(processing_first.status_code, 202)
        self.assertEqual(processing_second.status_code, 202)
        self.assertEqual(processing_first.json(), processing_second.json())
        self.assertIsNotNone(processing_after_first)
        self.assertIsNotNone(processing_after_second)
        self.assertEqual(processing_after_first["updated_at"], processing_after_second["updated_at"])

    def test_done_and_failed_tasks_are_not_cancellable(self) -> None:
        done_task = self._claim_processing_task()
        self.database.mark_task_done(done_task["id"], {"subtitle_paths": []})
        failed_task = self._claim_processing_task()
        self.database.mark_task_failure(failed_task["id"], "run_asr", "asr failed")

        with TestClient(create_app()) as client:
            done_response = client.post(f"/api/tasks/{done_task['id']}/cancel")
            failed_response = client.post(f"/api/tasks/{failed_task['id']}/cancel")

        self.assertEqual(done_response.status_code, 409)
        self.assertEqual(failed_response.status_code, 409)
        done_payload = done_response.json()
        failed_payload = failed_response.json()
        self.assertEqual(done_payload["outcome"], "not_cancellable")
        self.assertEqual(failed_payload["outcome"], "not_cancellable")
        self.assertIn("无法取消", done_payload["message"])
        self.assertIn("无法取消", failed_payload["message"])
        self.assertEqual(self.database.get_task(done_task["id"])["status"], "done")
        self.assertEqual(self.database.get_task(failed_task["id"])["status"], "failed")

    def test_done_transition_does_not_override_cancel_request(self) -> None:
        task = self._claim_processing_task()

        with TestClient(create_app()) as client:
            cancel_response = client.post(f"/api/tasks/{task['id']}/cancel")

        self.assertEqual(cancel_response.status_code, 202)
        self.assertTrue(self.database.is_cancel_requested(task["id"]))
        self.database.mark_task_done(task["id"], {"subtitle_paths": ["should-not-exist.srt"]})

        stored = self.database.get_task(task["id"])
        self.assertIsNotNone(stored)
        self.assertEqual(stored["status"], "cancelled")
        self.assertIsNone(stored["result_payload"])

    def test_default_task_list_includes_all_statuses_and_supports_filtering_and_pagination(self) -> None:
        tasks = [self._create_task() for _ in range(5)]
        statuses = ["pending", "processing", "failed", "done", "cancelled"]
        with self.database.connect() as connection:
            for task, status in zip(tasks, statuses):
                connection.execute(
                    "UPDATE tasks SET status = ?, finished_at = ? WHERE id = ?",
                    (status, "test" if status in {"failed", "done", "cancelled"} else None, task["id"]),
                )

        page = self.database.list_tasks(page=1, page_size=20)
        self.assertEqual({item["status"] for item in page.items}, set(statuses))
        self.assertEqual(page.total, 5)
        self.assertEqual(
            {item["status"] for item in self.database.list_tasks(page=1, page_size=20, status="done").items},
            {"done"},
        )

        with TestClient(create_app()) as client:
            response = client.get("/api/tasks")
            completed_response = client.get("/api/tasks", params={"status": "done"})
            pages = [
                client.get("/api/tasks", params={"page": number, "page_size": 2}).json()
                for number in range(1, 4)
            ]

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total"], 5)
        self.assertEqual(
            {item["status"] for item in response.json()["items"]},
            set(statuses),
        )
        self.assertEqual(response.json()["status_counts"], dict.fromkeys(statuses, 1))
        self.assertEqual(completed_response.status_code, 200)
        self.assertEqual(completed_response.json()["total"], 1)
        self.assertEqual(completed_response.json()["items"][0]["status"], "done")
        self.assertEqual([page["total"] for page in pages], [5, 5, 5])
        self.assertEqual([len(page["items"]) for page in pages], [2, 2, 1])
        self.assertEqual(
            [item["id"] for page in pages for item in page["items"]],
            [item["id"] for item in response.json()["items"]],
        )

    def test_failure_after_cancel_does_not_requeue_or_retry(self) -> None:
        task = self._claim_processing_task()

        with TestClient(create_app()) as client:
            cancel_response = client.post(f"/api/tasks/{task['id']}/cancel")

        self.assertEqual(cancel_response.status_code, 202)
        updated = self.database.mark_task_failure(
            task["id"],
            "translate",
            "provider disconnected while cancellation was pending",
        )

        self.assertEqual(updated["status"], "cancelled")
        self.assertEqual(updated["retry_count"], 0)
        self.assertIsNone(self.database.claim_next_pending_task())


if __name__ == "__main__":
    unittest.main()
