from __future__ import annotations

import json
import os
import signal
import sys
import tempfile
import threading
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from app.llm import LLM_EMPTY_RESPONSE, LLM_HTTP_ERROR, LLMError
from app.main import create_app
from app.runtime import WorkerService
from app.store import Database
from app.worker_control import WorkerIdentity, request_forced_worker_shutdown


WORKER_PID = 43210
WORKER_START_ID = "987654"
WORKER_COMMAND = "python3 -m app.worker_process"
SECRET_API_KEY = "sk-live-t7-integration-secret-123456"
SECRET_PROMPT = "T7_PROMPT_MUST_NOT_LEAK"


class BlockingTranslationProvider:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls: list[list[str]] = []

    def translate_batch(self, texts: list[str], target_language: str) -> list[str]:
        self.calls.append(list(texts))
        if len(self.calls) == 1:
            self.started.set()
            if not self.release.wait(5):
                raise AssertionError("test did not release the first translation chunk")
        return [f"{target_language}:{text}" for text in texts]


class FailingLLMClient:
    provider_name = "openai-compatible"

    def __init__(self, *, error_code: str, finish_reason: str | None = None, status_code: int | None = None) -> None:
        self.error_code = error_code
        self.finish_reason = finish_reason
        self.status_code = status_code

    def resolved_base_url(self) -> str:
        return "https://offline.invalid/v1"

    def complete(self, messages, cancellation_check=None) -> str:
        if cancellation_check is not None:
            cancellation_check()
        details = {
            "provider": self.provider_name,
            "model": "offline-test-model",
            "request_id": f"req-{self.error_code.lower()}",
            "api_key": SECRET_API_KEY,
            "prompt": SECRET_PROMPT,
            "response_preview": f"Bearer {SECRET_API_KEY}",
        }
        if self.finish_reason is not None:
            details["finish_reason"] = self.finish_reason
        if self.status_code is not None:
            details["status_code"] = self.status_code
        raise LLMError(
            f"offline translation failure: {SECRET_API_KEY}",
            details,
            error_code=self.error_code,
        )


class CancellationDiagnosticsIntegrationTests(unittest.TestCase):
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

        self.db_path = self.config_dir / "integration.db"
        self.database = Database(str(self.db_path))
        self.database.initialize()
        self.database.update_config(
            {
                "file": {
                    "input_dir": str(self.data_dir),
                    "output_to_source_dir": True,
                    "min_size_mb": 0,
                    "allowed_extensions": [".mp4"],
                },
                "processing": {
                    "work_dir": str(self.config_dir / "work"),
                    "max_retries": 0,
                    "keep_intermediates": True,
                },
                "translation": {
                    "enabled": True,
                    "target_languages": ["zh-CN"],
                    "max_retries": 1,
                    "timeout_seconds": 1,
                    "llm_type": "openai-compatible",
                    "api_base_url": "https://offline.invalid",
                    "api_key": SECRET_API_KEY,
                    "model": "offline-test-model",
                    "content_type": "general",
                    "custom_prompt": "",
                },
                "subtitle": {
                    "bilingual": False,
                    "bilingual_mode": "separate",
                    "filename_template": "{stem}.{lang}.srt",
                    "source_language": "en",
                },
                "mux": {"enabled": False},
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

    def _create_task(self, name: str = "video") -> dict:
        self.task_sequence += 1
        video_path = self.data_dir / f"{name}-{self.task_sequence}.mp4"
        video_path.write_bytes(b"offline-video")
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

    @staticmethod
    def _segments(count: int = 31) -> list[dict]:
        return [
            {"start": float(index), "end": float(index + 1), "text": f"line {index}"}
            for index in range(count)
        ]

    def _fast_pipeline_patches(self, segments: list[dict]) -> ExitStack:
        stack = ExitStack()

        def fake_extract_audio(context, cancel_token=None) -> Path:
            audio_path = Path(context.file_path).with_suffix(".wav")
            audio_path.write_bytes(b"RIFF")
            return audio_path

        stack.enter_context(patch("app.runtime.extract_audio", side_effect=fake_extract_audio))
        stack.enter_context(
            patch(
                "app.runtime.run_asr",
                return_value={
                    "segments": segments,
                    "language": "en",
                    "provider": "whisperx",
                    "device": "cpu",
                },
            )
        )
        stack.enter_context(patch("app.runtime.align_segments", return_value=segments))
        stack.enter_context(patch("app.runtime.process_text_segments", return_value=segments))
        return stack

    def _run_worker(self, worker: WorkerService) -> tuple[threading.Thread, dict[str, object]]:
        outcome: dict[str, object] = {}

        def run() -> None:
            try:
                outcome["processed"] = worker.process_next_task()
            except BaseException as exc:  # pragma: no cover - surfaced by the assertion below
                outcome["error"] = exc

        thread = threading.Thread(target=run, name="t7-worker", daemon=True)
        thread.start()
        return thread, outcome

    def _assert_secret_absent(self, serialized: str) -> None:
        self.assertNotIn(SECRET_API_KEY, serialized)
        self.assertNotIn(SECRET_PROMPT, serialized)
        self.assertNotIn("Bearer sk-live-t7", serialized)

    def test_pending_cancel_is_immediate_idempotent_and_retry_clears_cancel_state(self) -> None:
        task = self._create_task("pending-cancel")

        with TestClient(create_app()) as client:
            first = client.post(f"/api/tasks/{task['id']}/cancel")
            second = client.post(f"/api/tasks/{task['id']}/cancel")
            detail = client.get(f"/api/tasks/{task['id']}")
            logs = client.get(f"/api/tasks/{task['id']}/logs", params={"page": 1, "page_size": 50})
            self.assertIsNone(self.database.claim_next_pending_task())
            retried = client.post(f"/api/tasks/{task['id']}/retry", json={"mode": "restart"})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json()["outcome"], "cancelled")
        self.assertEqual(first.json()["status"], "cancelled")
        self.assertEqual(second.json(), first.json())
        self.assertTrue(detail.json()["cancel_requested"])
        self.assertIsNotNone(detail.json()["finished_at"])

        cancellation_logs = [
            item for item in logs.json()["items"] if item["message"] == "待处理任务已取消"
        ]
        self.assertEqual(len(cancellation_logs), 1)
        self.assertEqual(retried.status_code, 200)
        self.assertEqual(retried.json()["status"], "pending")
        self.assertFalse(retried.json()["cancel_requested"])
        stored = self.database.get_task(task["id"])
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertIsNone(stored["finished_at"])

    def test_processing_translation_cancel_stops_remaining_chunks_and_never_marks_done(self) -> None:
        task = self._create_task("translation-cancel")
        provider = BlockingTranslationProvider()
        worker = WorkerService(self.database)

        with self._fast_pipeline_patches(self._segments()), patch(
            "app.pipeline.get_translation_provider", return_value=provider
        ), TestClient(create_app()) as client:
            thread, outcome = self._run_worker(worker)
            self.assertTrue(provider.started.wait(5), "worker did not enter translation")
            cancel_response = client.post(f"/api/tasks/{task['id']}/cancel")
            self.assertEqual(cancel_response.status_code, 202)
            self.assertEqual(cancel_response.json()["outcome"], "cancelling")
            self.assertEqual(cancel_response.json()["status"], "processing")
            self.assertTrue(cancel_response.json()["cancel_requested"])

            provider.release.set()
            thread.join(5)
            self.assertFalse(thread.is_alive(), "worker did not stop after translation cancellation")
            self.assertNotIn("error", outcome)
            self.assertTrue(outcome.get("processed"))

            logs = client.get(f"/api/tasks/{task['id']}/logs", params={"page": 1, "page_size": 100})

        stored = self.database.get_task(task["id"])
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored["status"], "cancelled")
        self.assertEqual(stored["stage"], "translate")
        self.assertEqual(len(provider.calls), 1, "a subsequent translation chunk was started after cancellation")
        self.assertIsNone(stored["result_payload"])
        self.assertNotIn("任务处理完成", [item["message"] for item in logs.json()["items"]])

        # A late completion path must not overwrite the authoritative cancellation.
        self.database.mark_task_done(task["id"], {"subtitle_paths": ["should-not-exist.srt"]})
        final = self.database.get_task(task["id"])
        self.assertIsNotNone(final)
        assert final is not None
        self.assertEqual(final["status"], "cancelled")

    def test_non_cooperative_asr_cancel_signals_worker_and_restart_recovers_cancelled(self) -> None:
        task = self._create_task("forced-asr")
        claimed = self.database.claim_next_pending_task()
        self.assertIsNotNone(claimed)
        assert claimed is not None
        self.database.update_task_stage(task["id"], "run_asr", 35)
        self.database.register_worker(WORKER_PID, WORKER_START_ID, WORKER_COMMAND, WORKER_PID)
        self.database.update_worker_task(WORKER_PID, WORKER_START_ID, task["id"], "run_asr")
        identity = WorkerIdentity(WORKER_PID, WORKER_START_ID, WORKER_COMMAND, WORKER_PID)
        real_request = request_forced_worker_shutdown

        def request_without_background_kill(database: Database, task_id: int):
            return real_request(database, task_id, schedule_force_kill=False)

        with patch("app.worker_control.read_process_identity", return_value=identity), patch(
            "app.worker_control.os.killpg", create=True
        ) as killpg, patch("app.main.request_forced_worker_shutdown", side_effect=request_without_background_kill):
            with TestClient(create_app()) as client:
                first = client.post(f"/api/tasks/{task['id']}/cancel")
                second = client.post(f"/api/tasks/{task['id']}/cancel")
                self.assertEqual(first.status_code, 202)
                self.assertEqual(first.json()["outcome"], "cancelling")
                self.assertEqual(second.json(), first.json())

                # Simulate Supervisor restarting the Worker after SIGTERM.
                self.database.recover_orphaned_tasks()
                detail = client.get(f"/api/tasks/{task['id']}")
                logs = client.get(f"/api/tasks/{task['id']}/logs", params={"page": 1, "page_size": 100})

        killpg.assert_called_once_with(WORKER_PID, signal.SIGTERM)
        self.assertEqual(detail.json()["status"], "cancelled")
        self.assertEqual(detail.json()["stage"], "run_asr")
        self.assertIn("Worker 已终止", detail.json()["error_message"])
        messages = [item["message"] for item in logs.json()["items"]]
        self.assertIn("已请求强制终止 Worker", messages)
        self.assertIn("Worker 已终止，任务取消", messages)

    def test_llm_empty_filter_and_http_500_failures_keep_safe_diagnostics_in_api(self) -> None:
        cases = [
            ("empty", LLM_EMPTY_RESPONSE, "stop", None, {}),
            ("content-filter", LLM_EMPTY_RESPONSE, "content_filter", None, {"finish_reason": "content_filter"}),
            ("http-500", LLM_HTTP_ERROR, None, 500, {"http_status": 500}),
        ]

        for name, error_code, finish_reason, status_code, expected_details in cases:
            with self.subTest(case=name):
                task = self._create_task(name)
                client = FailingLLMClient(
                    error_code=error_code,
                    finish_reason=finish_reason,
                    status_code=status_code,
                )
                worker = WorkerService(self.database)
                with self._fast_pipeline_patches(self._segments(count=3)), patch(
                    "app.pipeline.create_llm_client", return_value=client
                ):
                    self.assertTrue(worker.process_next_task())

                with TestClient(create_app()) as api:
                    detail_response = api.get(f"/api/tasks/{task['id']}")
                    logs_response = api.get(
                        f"/api/tasks/{task['id']}/logs",
                        params={"page": 1, "page_size": 100},
                    )

                self.assertEqual(detail_response.status_code, 200)
                detail = detail_response.json()
                self.assertEqual(detail["status"], "failed")
                self.assertEqual(detail["stage"], "translate")
                self.assertNotIn("details_json", logs_response.json()["items"][0])

                error_log = next(item for item in logs_response.json()["items"] if item["level"] == "ERROR")
                details = error_log["details"]
                self.assertEqual(details["error_code"], error_code)
                self.assertEqual(details["provider"], "openai-compatible")
                self.assertEqual(details["model"], "offline-test-model")
                self.assertFalse(details["will_retry"])
                for key, value in expected_details.items():
                    self.assertEqual(details[key], value)

                serialized_detail = json.dumps(detail, ensure_ascii=False)
                serialized_logs = json.dumps(logs_response.json(), ensure_ascii=False)
                self._assert_secret_absent(serialized_detail)
                self._assert_secret_absent(serialized_logs)
                self._assert_secret_absent(json.dumps(error_log, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
