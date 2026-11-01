from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.llm import LLMMessage
from app.llm.providers import OpenAICompatibleLLMClient
from app.pipeline import (
    CancellationRequested,
    CancellationToken,
    ChunkedTranslator,
    LLMTranslationProvider,
    TaskContext,
    TranslationRateLimitError,
    TranslationProvider,
    extract_audio,
    mux_subtitle,
    translate_segments,
)
from app.runtime import WorkerService
from app.store import Database


class RecordingTranslationProvider(TranslationProvider):
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def translate_batch(self, texts: list[str], target_language: str) -> list[str]:
        self.calls.append((target_language, list(texts)))
        return [f"translated:{text}" for text in texts]


class RaisingCancellationProvider(TranslationProvider):
    def translate_batch(self, texts: list[str], target_language: str) -> list[str]:
        raise CancellationRequested("translate")


class RateLimitThenCancelProvider(TranslationProvider):
    def __init__(self, cancel_state: dict[str, bool]) -> None:
        self.cancel_state = cancel_state
        self.calls = 0

    def translate_batch(self, texts: list[str], target_language: str) -> list[str]:
        self.calls += 1
        self.cancel_state["cancelled"] = True
        raise TranslationRateLimitError("rate limited")


class CancellingLLMClient:
    def complete(self, messages, cancellation_check=None) -> str:
        raise CancellationRequested("translate")

    def resolved_base_url(self) -> str:
        return "http://example.invalid/v1"


class FakeFfmpegProcess:
    def __init__(self, *, terminate_hangs: bool = True) -> None:
        self.pid = 4242
        self.returncode: int | None = None
        self.terminate_hangs = terminate_hangs
        self.terminated = False
        self.killed = False
        self.communicate_calls = 0

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        self.communicate_calls += 1
        if self.communicate_calls == 1:
            raise subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=timeout)
        if self.terminated and not self.killed and self.terminate_hangs:
            raise subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=timeout)
        self.returncode = -9 if self.killed else -15
        return "", ""

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def poll(self) -> int | None:
        return self.returncode


class CooperativeCancellationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base = Path(self.temp_dir.name)
        self.data_dir = self.base / "data"
        self.output_dir = self.base / "output"
        self.work_dir = self.base / "work"
        for path in (self.data_dir, self.output_dir, self.work_dir):
            path.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _translation_context(self, max_retries: int = 4) -> TaskContext:
        snapshot = {
            "translation": {
                "enabled": True,
                "target_languages": ["zh-CN"],
                "max_retries": max_retries,
                "content_type": "general",
                "custom_prompt": "",
                "api_base_url": "http://example.invalid",
                "api_key": "test-key",
                "model": "fake-model",
                "timeout_seconds": 1,
            }
        }
        return TaskContext(
            task_id=1,
            file_path=str(self.data_dir / "source.mp4"),
            config_snapshot=snapshot,
            work_dir=self.work_dir,
        )

    def _ffmpeg_context(self) -> TaskContext:
        source_path = self.data_dir / "source.mp4"
        source_path.write_bytes(b"source-video")
        snapshot = {
            "whisper": {"audio_format": "wav", "sample_rate": 16000},
            "file": {"output_to_source_dir": True},
            "translation": {"enabled": False},
            "subtitle": {
                "bilingual": False,
                "bilingual_mode": "merge",
                "filename_template": "{stem}.{lang}.srt",
                "source_language": "en",
            },
            "mux": {"filename_template": "{stem}.subbed.mkv"},
        }
        return TaskContext(
            task_id=1,
            file_path=str(source_path),
            config_snapshot=snapshot,
            work_dir=self.work_dir,
        )

    def test_translation_stops_at_chunk_boundary_without_retry(self) -> None:
        cancelled = False
        token = CancellationToken(lambda: cancelled)
        provider = RecordingTranslationProvider()
        progress_events: list[tuple[int, int]] = []
        segments = [
            {"start": float(index), "end": float(index + 1), "text": f"line {index}"}
            for index in range(31)
        ]

        def on_progress(current: int, total: int) -> None:
            nonlocal cancelled
            progress_events.append((current, total))
            if current == 1:
                cancelled = True

        with patch("app.pipeline.get_translation_provider", return_value=provider):
            with self.assertRaises(CancellationRequested):
                translate_segments(
                    self._translation_context(max_retries=4),
                    segments,
                    progress_callback=on_progress,
                    cancel_token=token,
                )

        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(progress_events, [(1, 3)])

    def test_chunked_translator_propagates_cancellation_requested(self) -> None:
        translator = ChunkedTranslator(
            RaisingCancellationProvider(),
            "general",
            "",
            cancel_token=CancellationToken.never(),
        )

        with self.assertRaises(CancellationRequested):
            translator.translate_language(
                [{"start": 0.0, "end": 1.0, "text": "hello"}],
                "zh-CN",
            )

    def test_rate_limit_backoff_observes_cancellation(self) -> None:
        cancel_state = {"cancelled": False}
        provider = RateLimitThenCancelProvider(cancel_state)
        translator = ChunkedTranslator(
            provider,
            "general",
            "",
            cancel_token=CancellationToken(lambda: cancel_state["cancelled"]),
        )

        with self.assertRaises(CancellationRequested):
            translator.translate_language(
                [{"start": 0.0, "end": 1.0, "text": "hello"}],
                "zh-CN",
            )

        self.assertEqual(provider.calls, 1)

    def test_llm_provider_does_not_wrap_cancellation_requested(self) -> None:
        with patch("app.pipeline.create_llm_client", return_value=CancellingLLMClient()):
            provider = LLMTranslationProvider(
                llm_type="openai-compatible",
                api_base_url="http://example.invalid",
                api_key="test-key",
                model="fake-model",
                timeout_seconds=1,
            )

        with self.assertRaises(CancellationRequested):
            provider.translate_batch(["hello"], "zh-CN")

    @patch("app.llm.providers.OpenAI")
    def test_openai_stream_checks_cancellation_between_chunks(self, mock_openai: MagicMock) -> None:
        yielded: list[int] = []

        def stream():
            for index in range(3):
                yielded.append(index)
                yield SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=str(index)),
                            finish_reason=None,
                        )
                    ]
                )

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = stream()
        mock_openai.return_value = mock_client
        client = OpenAICompatibleLLMClient(
            "http://example.invalid",
            "test-key",
            "fake-model",
            1,
        )
        checks = 0

        def cancellation_check() -> None:
            nonlocal checks
            checks += 1
            if checks == 3:
                raise CancellationRequested("translate")

        with self.assertRaises(CancellationRequested):
            client.complete(
                [LLMMessage(role="user", content="hello")],
                cancellation_check=cancellation_check,
            )

        self.assertEqual(yielded, [0, 1])

    def test_extract_audio_cancellation_terminates_ffmpeg(self) -> None:
        context = self._ffmpeg_context()
        process = FakeFfmpegProcess()
        started = False

        def start_process(*args, **kwargs):
            nonlocal started
            started = True
            return process

        token = CancellationToken(lambda: started)
        with patch("app.pipeline.shutil.which", return_value="ffmpeg"), patch(
            "app.pipeline.subprocess.Popen",
            side_effect=start_process,
        ):
            with self.assertRaises(CancellationRequested):
                extract_audio(context, token)

        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertEqual(process.communicate_calls, 3)

    def test_mux_cancellation_terminates_ffmpeg(self) -> None:
        context = self._ffmpeg_context()
        subtitle_path = self.output_dir / "source.en.srt"
        subtitle_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
        process = FakeFfmpegProcess()
        started = False

        def start_process(*args, **kwargs):
            nonlocal started
            started = True
            return process

        token = CancellationToken(lambda: started)
        with patch("app.pipeline.shutil.which", return_value="ffmpeg"), patch(
            "app.pipeline.subprocess.Popen",
            side_effect=start_process,
        ):
            with self.assertRaises(CancellationRequested):
                mux_subtitle(context, [str(subtitle_path)], token)

        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertEqual(process.communicate_calls, 3)

    def test_worker_marks_cancellation_cancelled_without_retry(self) -> None:
        database = Database(str(self.base / "test.db"))
        database.initialize()
        database.update_config(
            {
                "processing": {
                    "work_dir": str(self.work_dir),
                    "max_retries": 2,
                }
            }
        )
        source_path = self.data_dir / "cancel.mp4"
        source_path.write_bytes(b"video")
        observed = database.observe_file(str(source_path), source_path.stat().st_size, source_path.stat().st_mtime)
        task = database.create_task(
            observed["file_id"],
            observed["path"],
            observed["size_bytes"],
            observed["mtime"],
        )
        worker = WorkerService(database)

        def cancel_during_extract(context: TaskContext, cancel_token: CancellationToken) -> Path:
            database.request_cancel(task["id"])
            cancel_token.raise_if_cancelled("extract_audio")
            raise AssertionError("cancellation token did not observe the database state")

        with patch("app.runtime.extract_audio", side_effect=cancel_during_extract):
            self.assertTrue(worker.process_next_task())

        cancelled_task = database.get_task(task["id"])
        self.assertIsNotNone(cancelled_task)
        assert cancelled_task is not None
        self.assertEqual(cancelled_task["status"], "cancelled")
        self.assertEqual(cancelled_task["stage"], "extract_audio")
        self.assertEqual(cancelled_task["retry_count"], 0)


if __name__ == "__main__":
    unittest.main()
