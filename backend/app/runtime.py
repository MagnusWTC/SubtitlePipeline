from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .llm import LLMError, sanitize_diagnostic_data
from .pipeline import (
    CancellationRequested,
    CancellationToken,
    PipelineError,
    TaskContext,
    align_segments,
    build_result_payload,
    cleanup_intermediates,
    cleanup_work_dir_intermediates,
    ensure_intermediates_dir,
    extract_audio,
    load_asr_result,
    load_aligned_segments,
    load_processed_segments,
    load_translations,
    mux_subtitle,
    normalize_stage_name,
    process_text_segments,
    read_stage_artifacts,
    render_srt,
    resolve_audio_path,
    run_asr,
    save_aligned_segments,
    save_asr_result,
    save_processed_segments,
    save_translations,
    translate_segments,
    write_stage_artifacts,
    WhisperModelCache,
)
from .store import Database
from .worker_control import WorkerIdentity


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".wmv", ".m4v"}
logger = logging.getLogger(__name__)

_MAX_DIAGNOSTIC_DEPTH = 4
_MAX_DIAGNOSTIC_ITEMS = 20
_MAX_DIAGNOSTIC_STRING = 300
_SENSITIVE_DIAGNOSTIC_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "body",
    "content",
    "exception_message",
    "input",
    "lines",
    "message",
    "messages",
    "payload",
    "prompt",
    "raw",
    "raw_body",
    "refusal_preview",
    "request_body",
    "response_body",
    "response_preview",
    "segments",
    "source_text",
    "subtitle",
    "subtitles",
    "system_prompt",
    "text",
    "transcript",
    "upstream_body_preview",
}


def _normalized_diagnostic_key(key: str) -> str:
    return key.strip().lower().replace("-", "_").replace(" ", "_")


def _is_sensitive_diagnostic_key(key: str) -> bool:
    normalized = _normalized_diagnostic_key(key)
    return (
        normalized in _SENSITIVE_DIAGNOSTIC_KEYS
        or normalized.endswith("_api_key")
        or normalized.endswith("_authorization")
        or normalized.endswith("_credential")
        or normalized.endswith("_password")
        or normalized.endswith("_preview")
        or normalized.endswith("_secret")
        or normalized.endswith("_token")
    )


def _limit_diagnostic_value(value: Any, depth: int = 0) -> Any:
    if depth >= _MAX_DIAGNOSTIC_DEPTH:
        return "[MAX_DEPTH]"
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in list(value.items())[:_MAX_DIAGNOSTIC_ITEMS]:
            key_text = str(key)
            if _is_sensitive_diagnostic_key(key_text):
                continue
            result[key_text] = _limit_diagnostic_value(item, depth + 1)
        return result
    if isinstance(value, list):
        return [_limit_diagnostic_value(item, depth + 1) for item in value[:_MAX_DIAGNOSTIC_ITEMS]]
    if isinstance(value, str):
        return value if len(value) <= _MAX_DIAGNOSTIC_STRING else f"{value[:_MAX_DIAGNOSTIC_STRING]}..."
    return value


def _extract_failure_diagnostics(exc: BaseException) -> dict[str, Any]:
    seen: set[int] = set()
    current: BaseException | None = exc
    diagnostic_source: BaseException | None = None
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        details = getattr(current, "details", None)
        if isinstance(current, LLMError) or (isinstance(details, dict) and details):
            diagnostic_source = current
            break
        current = current.__cause__ or current.__context__

    if diagnostic_source is None:
        return {}

    sanitized = sanitize_diagnostic_data(getattr(diagnostic_source, "details", None) or {})
    details = _limit_diagnostic_value(sanitized)
    if not isinstance(details, dict):
        details = {}
    error_code = getattr(diagnostic_source, "error_code", None) or details.get("error_code")
    if error_code:
        details["error_code"] = str(error_code)
    status_code = details.get("http_status", details.get("status_code"))
    try:
        if status_code is not None:
            details["http_status"] = int(status_code)
    except (TypeError, ValueError):
        pass
    return details


def _safe_failure_message(exc: BaseException) -> str:
    sanitized = sanitize_diagnostic_data(str(exc) or type(exc).__name__)
    message = sanitized if isinstance(sanitized, str) else type(exc).__name__
    return message[:_MAX_DIAGNOSTIC_STRING]


def _diagnostic_summary(details: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("error_code", "http_status", "finish_reason", "provider", "model", "retry_mode"):
        value = details.get(key)
        if value is not None and value != "":
            parts.append(f"{key}={value}")
    return f" ({', '.join(parts)})" if parts else ""


def _log_worker_failure(task_id: int, stage: str, task: dict[str, Any], details: dict[str, Any]) -> None:
    summary = _diagnostic_summary(details)
    if task.get("status") == "pending":
        logger.warning(
            "任务 %s 在阶段 %s 失败，将自动重试 (%s/%s)%s",
            task_id,
            stage,
            task.get("retry_count"),
            task.get("max_retries"),
            summary,
        )
    else:
        logger.error("任务 %s 在阶段 %s 最终失败%s", task_id, stage, summary)


def _mux_output_suffix(config: dict) -> str:
    """Return the fixed suffix of mux output filenames (e.g. '.subbed.mkv')."""
    template = str(config.get("mux", {}).get("filename_template", "")).strip()
    if not template:
        return ""
    suffix = template.replace("{stem}", "")
    return suffix if suffix else ""


def _is_inside_directory(path: Path, directory: Path) -> bool:
    try:
        return path.resolve().is_relative_to(directory.resolve())
    except OSError:
        return False


def _should_skip_scan_path(path: Path, config: dict[str, Any]) -> bool:
    if ".subpipeline" in path.parts:
        return True
    if not bool(config.get("file", {}).get("output_to_source_dir", True)):
        output_dir = Path(os.environ.get("SUBPIPELINE_OUTPUT_DIR", "/output"))
        if _is_inside_directory(path, output_dir):
            return True
    mux_suffix = _mux_output_suffix(config)
    return bool(mux_suffix and path.name.endswith(mux_suffix))


@dataclass
class ScanResult:
    scanned: int
    queued: int
    skipped: int
    pending_count: int
    remaining_slots: int
    throttled: bool


class ScannerService:
    def __init__(self, database: Database):
        self.database = database

    def scan_once(self) -> ScanResult:
        config = self.database.get_config()
        file_config = config["file"]
        input_dir = Path(file_config["input_dir"])
        input_dir.mkdir(parents=True, exist_ok=True)
        allowed = {extension.lower() for extension in file_config.get("allowed_extensions", [])} or VIDEO_EXTENSIONS
        min_size_bytes = int(file_config["min_size_mb"]) * 1024 * 1024
        max_size_bytes = int(file_config["max_size_mb"]) * 1024 * 1024
        pending_count = self.database.count_tasks_by_status("pending")
        scanned = 0
        queued = 0
        skipped = 0
        # 收集所有文件，按文件夹创建时间（新→旧）和路径深度（外→内）排序
        all_files = [p for p in input_dir.rglob("*") if p.is_file()]

        @lru_cache(maxsize=None)
        def _dir_ctime(d: str) -> float:
            try:
                return os.stat(d).st_ctime
            except OSError:
                return 0.0

        input_depth = len(input_dir.parts)

        def _sort_key(p: Path):
            depth = len(p.parts) - input_depth - 1          # 外层优先 (小 → 大)
            parent_ctime = _dir_ctime(str(p.parent))        # 新建优先 (大 → 小)
            return (depth, -parent_ctime, p.name)

        all_files.sort(key=_sort_key)

        for path in all_files:
            if _should_skip_scan_path(path, config):
                continue
            if path.suffix.lower() not in allowed:
                continue
            scanned += 1
            stat = path.stat()
            observed = self.database.observe_file(str(path), int(stat.st_size), float(stat.st_mtime))
            if observed["size_bytes"] < min_size_bytes or observed["size_bytes"] > max_size_bytes:
                skipped += 1
                continue
            if observed["stable_hits"] < 2:
                skipped += 1
                continue
            if self.database.has_task_for_file_version(
                observed["path_key"],
                observed["size_bytes"],
                observed["mtime"],
            ):
                skipped += 1
                continue
            if self.database.has_active_task(observed["path_key"]):
                skipped += 1
                continue
            parent_dir_ctime = _dir_ctime(str(path.parent))
            self.database.create_task(
                observed["file_id"],
                observed["path"],
                observed["size_bytes"],
                observed["mtime"],
                parent_dir_ctime,
            )
            queued += 1
        return ScanResult(
            scanned=scanned,
            queued=queued,
            skipped=skipped,
            pending_count=pending_count,
            remaining_slots=0,
            throttled=False,
        )

    def run_forever(self) -> None:
        while True:
            if not self.database.is_setup_complete():
                interval = int(self.database.get_config()["file"]["scan_interval_seconds"])
                time.sleep(max(interval, 1))
                continue
            result = self.scan_once()
            logger.info(
                "扫描完成 scanned=%d queued=%d skipped=%d pending=%d slots=%d throttled=%s",
                result.scanned,
                result.queued,
                result.skipped,
                result.pending_count,
                result.remaining_slots,
                result.throttled,
            )
            interval = int(self.database.get_config()["file"]["scan_interval_seconds"])
            time.sleep(max(interval, 1))


class WorkerService:
    def __init__(self, database: Database, worker_identity: WorkerIdentity | None = None):
        self.database = database
        self.worker_identity = worker_identity
        self.model_cache = WhisperModelCache()

    def _set_worker_task(self, task_id: int | None, stage: str | None) -> None:
        if self.worker_identity is None:
            return
        self.database.update_worker_task(
            self.worker_identity.pid,
            self.worker_identity.process_start_id,
            task_id,
            stage,
        )

    def run_forever(self) -> None:
        while True:
            if not self.database.is_setup_complete():
                interval = int(self.database.get_config()["processing"]["poll_interval_seconds"])
                time.sleep(max(interval, 1))
                continue
            processed = self.process_next_task()
            if not processed:
                interval = int(self.database.get_config()["processing"]["poll_interval_seconds"])
                time.sleep(max(interval, 1))

    def process_next_task(self) -> bool:
        task = self.database.claim_next_pending_task()
        if not task:
            return False
        self._set_worker_task(int(task["id"]), str(task["stage"]))
        try:
            result_payload = self._process_claimed_task(task)
            self.database.mark_task_done(task["id"], result_payload)
        except CancellationRequested:
            latest = self.database.get_task(task["id"])
            self.database.mark_task_cancelled(task["id"], latest["stage"] if latest else task["stage"])
        except Exception as exc:
            latest = self.database.get_task(task["id"])
            failure_stage = latest["stage"] if latest else task["stage"]
            diagnostics = _extract_failure_diagnostics(exc)
            safe_message = _safe_failure_message(exc)
            updated = self.database.mark_task_failure(task["id"], failure_stage, safe_message, diagnostics)
            _log_worker_failure(task["id"], failure_stage, updated, diagnostics)
        finally:
            self._set_worker_task(None, None)
        return True

    def _process_claimed_task(self, task: dict[str, Any]) -> dict[str, Any]:
        snapshot = task["config_snapshot"]
        if snapshot is None:
            raise PipelineError("缺少配置快照")
        work_dir = Path(snapshot["processing"]["work_dir"]) / str(task["id"])
        work_dir.mkdir(parents=True, exist_ok=True)
        context = TaskContext(
            task_id=task["id"],
            file_path=task["file_path"],
            config_snapshot=snapshot,
            work_dir=work_dir,
        )
        task_id = int(task["id"])
        cancel_token = CancellationToken(lambda: self.database.is_cancel_requested(task_id))
        intermediates_dir = ensure_intermediates_dir(context)
        if context.using_fallback_intermediates:
            self.database.log(
                task["id"],
                task["stage"],
                "WARNING",
                "源文件目录不可写，已回退到工作目录保存中间产物",
                {"intermediates_dir": str(intermediates_dir)},
            )
        start_stage = normalize_stage_name(str(task["stage"]))
        audio_path: Path | None = None
        asr_result: dict[str, Any] | None = None
        aligned_segments: list[dict[str, Any]] | None = None
        processed_segments: list[dict[str, Any]] | None = None
        translations: dict[str, list[str]] | None = None
        subtitle_paths: list[str] | None = None
        result_payload: dict[str, Any] | None = None
        if start_stage not in {"queued", "extract_audio"}:
            audio_path = resolve_audio_path(context)
        if start_stage in {"align_segments", "text_process", "translate", "subtitle_render", "output_finalize", "mux"}:
            asr_result = load_asr_result(context)
        if start_stage in {"text_process", "translate", "subtitle_render", "output_finalize", "mux"}:
            aligned_segments = load_aligned_segments(context)
        if start_stage in {"translate", "subtitle_render", "output_finalize", "mux"}:
            processed_segments = load_processed_segments(context)
        if snapshot["translation"]["enabled"] and start_stage in {"subtitle_render", "output_finalize", "mux"}:
            translations = load_translations(context)
        if start_stage in {"output_finalize", "mux"}:
            if processed_segments is None:
                raise PipelineError("缺少字幕渲染依赖，无法继续执行")
            subtitle_paths = render_srt(context, processed_segments, translations or {})
        if start_stage == "mux":
            result_payload = read_stage_artifacts(context)
        if start_stage in {"queued", "extract_audio"}:
            audio_path = self._run_stage(
                task_id,
                "extract_audio",
                10,
                lambda: extract_audio(context, cancel_token),
                cancel_token,
            )
        if start_stage in {"queued", "extract_audio", "run_asr"}:
            if audio_path is None:
                raise PipelineError("缺少音频文件，无法执行 ASR")
            asr_result = self._run_stage(task["id"], "run_asr", 35, lambda: run_asr(context, audio_path, self.model_cache, self.database))
            save_asr_result(context, asr_result)
        if start_stage in {"queued", "extract_audio", "run_asr", "align_segments"}:
            if asr_result is None or audio_path is None:
                raise PipelineError("缺少 ASR 结果，无法执行时间轴对齐")
            aligned_segments = self._run_stage(
                task["id"],
                "align_segments",
                45,
                lambda: align_segments(context, asr_result, audio_path, self.model_cache, self.database),
            )
            save_aligned_segments(context, aligned_segments)
        if start_stage in {"queued", "extract_audio", "run_asr", "align_segments", "text_process"}:
            if aligned_segments is None:
                raise PipelineError("缺少对齐结果，无法继续执行文本处理")
            processed_segments = self._run_stage(
                task["id"],
                "text_process",
                55,
                lambda: process_text_segments(aligned_segments),
            )
            save_processed_segments(context, processed_segments)
        if start_stage in {"queued", "extract_audio", "run_asr", "align_segments", "text_process", "translate"}:
            if processed_segments is None:
                raise PipelineError("缺少分段结果，无法继续执行")
            translations = self._run_stage(
                task_id,
                "translate",
                60,
                lambda: translate_segments(
                    context,
                    processed_segments,
                    progress_callback=lambda current, total: self.database.update_task_stage(
                        task["id"],
                        "translate",
                        60 + int(20 * current / total),
                    ),
                    cancel_token=cancel_token,
                ),
                cancel_token,
            )
            save_translations(context, translations)
        if start_stage in {"queued", "extract_audio", "run_asr", "align_segments", "text_process", "translate", "subtitle_render"}:
            if processed_segments is None:
                raise PipelineError("缺少字幕渲染输入，无法继续执行")
            subtitle_paths = self._run_stage(
                task["id"],
                "subtitle_render",
                95,
                lambda: render_srt(context, processed_segments, translations or {}),
            )
        if audio_path is None or subtitle_paths is None:
            raise PipelineError("缺少最终输出所需产物")
        if start_stage in {"queued", "extract_audio", "run_asr", "align_segments", "text_process", "translate", "subtitle_render", "output_finalize"}:
            result_payload = build_result_payload(context, audio_path, subtitle_paths, translations or {})
            self._run_stage(task["id"], "output_finalize", 100, lambda: write_stage_artifacts(context, result_payload))
        if result_payload is None:
            raise PipelineError("缺少任务结果元数据")
        if snapshot["mux"]["enabled"]:
            mux_path = self._run_stage(
                task_id,
                "mux",
                100,
                lambda: mux_subtitle(context, subtitle_paths, cancel_token),
                cancel_token,
            )
            result_payload["mux_path"] = mux_path
            write_stage_artifacts(context, result_payload)
        if not snapshot["processing"]["keep_intermediates"]:
            cleanup_intermediates(Path(task["file_path"]))
            if context.using_fallback_intermediates:
                cleanup_work_dir_intermediates(context.work_dir)
        return result_payload

    def _run_stage(
        self,
        task_id: int,
        stage: str,
        progress: float,
        action,
        cancel_token: CancellationToken | None = None,
    ):
        self._ensure_not_cancelled(task_id, stage, cancel_token)
        self.database.update_task_stage(task_id, stage, progress)
        self._set_worker_task(task_id, stage)
        self.database.log(task_id, stage, "INFO", f"开始阶段 {stage}")
        try:
            result = action()
            self._ensure_not_cancelled(task_id, stage, cancel_token)
        except CancellationRequested:
            self.database.log(task_id, stage, "WARNING", f"阶段 {stage} 收到取消请求并停止")
            raise
        self.database.log(task_id, stage, "INFO", f"完成阶段 {stage}")
        return result

    def _ensure_not_cancelled(
        self,
        task_id: int,
        stage: str,
        cancel_token: CancellationToken | None = None,
    ) -> None:
        if cancel_token is not None:
            cancel_token.raise_if_cancelled(stage)
        elif self.database.is_cancel_requested(task_id):
            raise CancellationRequested(stage)
