from __future__ import annotations

import json
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

from app.llm import (
    LLM_HTTP_ERROR,
    LLM_RATE_LIMIT,
    LLMError,
)
from app.main import create_app
from app.runtime import WorkerService
from app.store import Database


class TaskLogDiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base = Path(self.temp_dir.name)
        self.data_dir = self.base / "data"
        self.output_dir = self.base / "output"
        self.config_dir = self.base / "config"
        self.models_dir = self.base / "models"
        self.frontend_dist = self.base / "frontend-dist"
        for path in (self.data_dir, self.output_dir, self.config_dir, self.models_dir, self.frontend_dist):
            path.mkdir(parents=True, exist_ok=True)
        (self.frontend_dist / "index.html").write_text("<!doctype html><title>SubPipeline</title>", encoding="utf-8")

        self.database = Database(str(self.config_dir / "subpipeline.db"))
        self.database.initialize()
        self.database.update_config(
            {
                "file": {
                    "input_dir": str(self.data_dir),
                    "output_to_source_dir": False,
                    "min_size_mb": 0,
                    "allowed_extensions": [".mp4"],
                },
                "processing": {
                    "work_dir": str(self.config_dir / "work"),
                    "max_retries": 1,
                    "retry_mode": "restart",
                },
                "translation": {"enabled": False},
            }
        )

        self.previous_env = {
            "SUBPIPELINE_DB_PATH": os.environ.get("SUBPIPELINE_DB_PATH"),
            "SUBPIPELINE_FRONTEND_DIST": os.environ.get("SUBPIPELINE_FRONTEND_DIST"),
            "SUBPIPELINE_MODELS_DIR": os.environ.get("SUBPIPELINE_MODELS_DIR"),
            "SUBPIPELINE_BROWSE_ROOTS": os.environ.get("SUBPIPELINE_BROWSE_ROOTS"),
            "SUBPIPELINE_OUTPUT_DIR": os.environ.get("SUBPIPELINE_OUTPUT_DIR"),
        }
        os.environ["SUBPIPELINE_DB_PATH"] = str(self.config_dir / "subpipeline.db")
        os.environ["SUBPIPELINE_FRONTEND_DIST"] = str(self.frontend_dist)
        os.environ["SUBPIPELINE_MODELS_DIR"] = str(self.models_dir)
        os.environ["SUBPIPELINE_BROWSE_ROOTS"] = ",".join(
            [str(self.data_dir), str(self.output_dir), str(self.config_dir)]
        )
        os.environ["SUBPIPELINE_OUTPUT_DIR"] = str(self.output_dir)

    def tearDown(self) -> None:
        for key, value in self.previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.temp_dir.cleanup()

    def _create_task(self, *, name: str = "diagnostics", stage: str = "translate") -> dict:
        video_path = self.data_dir / f"{name}.mp4"
        video_path.write_bytes(f"video-{name}".encode())
        observed = self.database.observe_file(
            str(video_path),
            int(video_path.stat().st_size),
            float(video_path.stat().st_mtime),
        )
        task = self.database.create_task(
            observed["file_id"],
            observed["path"],
            observed["size_bytes"],
            observed["mtime"],
        )
        self._set_stage(task["id"], stage)
        return task

    def _set_stage(self, task_id: int, stage: str) -> None:
        with self.database.connect() as connection:
            connection.execute(
                "UPDATE tasks SET stage = ?, updated_at = ? WHERE id = ?",
                (stage, "test", task_id),
            )

    def _worker(self) -> WorkerService:
        return WorkerService(self.database)

    def test_task_logs_api_returns_persisted_llm_diagnostics(self) -> None:
        self.database.update_config({"processing": {"max_retries": 0}})
        task = self._create_task(name="api-details")
        error = LLMError(
            "translation upstream rejected request",
            {
                "status_code": 400,
                "finish_reason": "length",
                "request_id": "req-diagnostic-1",
                "chunk_start_index": 12,
            },
            error_code=LLM_HTTP_ERROR,
        )

        def fail_translation(*_args: object) -> None:
            try:
                raise error
            except LLMError as cause:
                raise RuntimeError("translation wrapper") from cause

        worker = self._worker()
        with patch.object(worker, "_process_claimed_task", side_effect=fail_translation):
            with self.assertLogs("app.runtime", level="ERROR") as captured:
                self.assertTrue(worker.process_next_task())
        self.assertEqual(len(captured.records), 1)
        self.assertIn(LLM_HTTP_ERROR, captured.records[0].getMessage())

        app = create_app()
        with TestClient(app) as client:
            response = client.get(f"/api/tasks/{task['id']}/logs", params={"page": 1, "page_size": 20})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["page"], 1)
        self.assertEqual(payload["page_size"], 20)
        failure = next(item for item in payload["items"] if item["level"] == "ERROR")
        details = failure["details"]
        self.assertEqual(details["error_code"], LLM_HTTP_ERROR)
        self.assertEqual(details["status_code"], 400)
        self.assertEqual(details["http_status"], 400)
        self.assertEqual(details["finish_reason"], "length")
        self.assertEqual(details["request_id"], "req-diagnostic-1")
        self.assertEqual(details["chunk_start_index"], 12)
        self.assertFalse(details["will_retry"])
        self.assertEqual(details["retry_mode"], "restart")

    def test_retry_and_final_failure_keep_business_and_retry_diagnostics(self) -> None:
        task = self._create_task(name="retry-diagnostics")
        first_error = LLMError(
            "translation rate limited",
            {
                "status_code": 429,
                "finish_reason": "length",
                "chunk_start_index": 3,
            },
            error_code=LLM_RATE_LIMIT,
        )
        worker = self._worker()
        with patch.object(worker, "_process_claimed_task", side_effect=first_error):
            with self.assertLogs("app.runtime", level="WARNING") as retry_logs:
                self.assertTrue(worker.process_next_task())

        self.assertEqual(len(retry_logs.records), 1)
        retried = self.database.get_task(task["id"])
        assert retried is not None
        self.assertEqual(retried["status"], "pending")
        retry_details = self.database.get_logs(task["id"], page=1, page_size=20).items[0]["details"]
        self.assertEqual(retry_details["error_code"], LLM_RATE_LIMIT)
        self.assertEqual(retry_details["status_code"], 429)
        self.assertEqual(retry_details["http_status"], 429)
        self.assertEqual(retry_details["finish_reason"], "length")
        self.assertTrue(retry_details["will_retry"])
        self.assertEqual(retry_details["retry_mode"], "restart")

        self._set_stage(task["id"], "translate")
        final_error = LLMError(
            "translation failed permanently",
            {"status_code": 500, "request_id": "req-final"},
            error_code=LLM_HTTP_ERROR,
        )
        with patch.object(worker, "_process_claimed_task", side_effect=final_error):
            with self.assertLogs("app.runtime", level="ERROR") as final_logs:
                self.assertTrue(worker.process_next_task())

        self.assertEqual(len(final_logs.records), 1)
        failed = self.database.get_task(task["id"])
        assert failed is not None
        self.assertEqual(failed["status"], "failed")
        final_details = self.database.get_logs(task["id"], page=1, page_size=20).items[0]["details"]
        self.assertEqual(final_details["error_code"], LLM_HTTP_ERROR)
        self.assertEqual(final_details["status_code"], 500)
        self.assertEqual(final_details["http_status"], 500)
        self.assertEqual(final_details["request_id"], "req-final")
        self.assertFalse(final_details["will_retry"])
        self.assertEqual(final_details["retry_mode"], "restart")

    def test_failure_diagnostics_redact_sensitive_and_large_payloads(self) -> None:
        self.database.update_config({"processing": {"max_retries": 0}})
        task = self._create_task(name="redaction")
        prompt = "PROMPT_SHOULD_NOT_APPEAR"
        subtitle = "SUBTITLE_SHOULD_NOT_APPEAR"
        api_key = "sk-live-diagnostic-secret-123456"
        error = LLMError(
            "translation failed safely",
            {
                "status_code": 401,
                "error_code": LLM_HTTP_ERROR,
                "api_key": api_key,
                "prompt": prompt,
                "subtitles": [subtitle],
                "response_preview": subtitle * 40,
                "nested_sensitive": {
                    "authorization": "Bearer should-not-appear",
                    "prompt": prompt,
                },
                "safe_structured": {
                    "retryable": True,
                    "request_id": "req-safe",
                },
            },
            error_code=LLM_HTTP_ERROR,
        )

        worker = self._worker()
        with patch.object(worker, "_process_claimed_task", side_effect=error):
            with self.assertLogs("app.runtime", level="ERROR"):
                self.assertTrue(worker.process_next_task())

        failure = next(
            item
            for item in self.database.get_logs(task["id"], page=1, page_size=20).items
            if item["level"] == "ERROR"
        )
        serialized = json.dumps(failure, ensure_ascii=False)
        self.assertNotIn(prompt, serialized)
        self.assertNotIn(subtitle, serialized)
        self.assertNotIn(api_key, serialized)
        self.assertNotIn("should-not-appear", serialized)
        self.assertEqual(failure["details"]["error_code"], LLM_HTTP_ERROR)
        self.assertEqual(failure["details"]["http_status"], 401)
        self.assertEqual(failure["details"]["safe_structured"]["request_id"], "req-safe")

    def test_log_pagination_is_unchanged_when_details_are_present(self) -> None:
        task = self._create_task(name="pagination")
        for index in range(3):
            self.database.log(
                task["id"],
                "translate",
                "INFO",
                f"detail log {index}",
                {"sequence": index, "error_code": LLM_HTTP_ERROR},
            )

        first_page = self.database.get_logs(task["id"], page=1, page_size=1)
        second_page = self.database.get_logs(task["id"], page=2, page_size=1)

        self.assertEqual(first_page.total, 4)
        self.assertEqual(len(first_page.items), 1)
        self.assertEqual(first_page.items[0]["details"]["sequence"], 2)
        self.assertEqual(second_page.items[0]["details"]["sequence"], 1)


if __name__ == "__main__":
    unittest.main()
