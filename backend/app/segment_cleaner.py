"""Conservative subtitle cleanup using alignment timestamps when available."""

import math
import re
from typing import Any


def fix_timestamp_anomalies(segment: dict[str, Any]) -> dict[str, Any]:
    start = round(float(segment["start"]), 2)
    end = round(float(segment["end"]), 2)
    return {**segment, "start": start, "end": end if end > start else round(start + 0.1, 2)}


def compress_repetitions(text: str) -> str:
    # Only collapse recognizable vocalizations; ordinary repeated words may be intentional.
    text = re.sub(r"([あアぁうウぅえエぇおオぉんン啊呀嗯唔])\1{3,}", r"\1ー", text)
    text = re.sub(r"(えっ|あっ|なに|何)(?:\1){3,}", r"\1", text)
    return text


def _timed_words(segment: dict[str, Any]) -> list[tuple[int, int, float, float]]:
    text = str(segment["text"])
    words = segment.get("words")
    if not isinstance(words, list) or not words:
        return []
    result = []
    cursor = 0
    previous_end = float(segment["start"])
    for item in words:
        if not isinstance(item, dict):
            return []
        word = str(item.get("word", "")).strip()
        if not word or item.get("start") is None or item.get("end") is None:
            return []
        try:
            start, end = float(item["start"]), float(item["end"])
        except (TypeError, ValueError):
            return []
        position = text.find(word, cursor)
        if (position < 0 or text[cursor:position].strip() or not math.isfinite(start)
                or not math.isfinite(end) or start < previous_end - 0.05 or end < start
                or start < float(segment["start"]) - 0.1 or end > float(segment["end"]) + 0.1):
            return []
        cursor = position + len(word)
        result.append((position, cursor, start, end))
        previous_end = end
    if text[cursor:].strip():
        return []
    return result


def split_long_segment(segment: dict[str, Any], max_duration: float = 7.0) -> list[dict[str, Any]]:
    """Split at timed word boundaries; keep the original text if timing is incomplete."""
    words = _timed_words(segment)
    if len(words) < 2:
        return [segment]
    text = str(segment["text"])
    groups = []
    first = 0
    for i in range(len(words) - 1):
        _, end_pos, _, end = words[i]
        next_start = words[i + 1][2]
        punctuation = text[words[i][0]:end_pos].rstrip().endswith(("。", "！", "？", ".", "!", "?"))
        if punctuation or next_start - end >= 0.6 or words[i + 1][3] - words[first][2] > max_duration:
            groups.append((first, i))
            first = i + 1
    groups.append((first, len(words) - 1))
    if len(groups) == 1:
        return [segment]
    parts = []
    for first, last in groups:
        start_pos = words[first][0]
        end_pos = words[last][1]
        parts.append({"start": round(words[first][2], 2), "end": round(words[last][3], 2),
                      "text": text[start_pos:end_pos].strip()})
    return [fix_timestamp_anomalies(part) for part in parts]


def _vocalization(text: str) -> str | None:
    text = text.strip().rstrip("。.!！?？、,，")
    return text if re.fullmatch(r"[あアぁうウぅえエぇおオぉんン啊呀嗯唔ー〜～っッ]+", text) else None


def clean_segments(segments: list[dict[str, Any]], max_duration: float = 7.0, max_gap: float = 0.5) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for original in segments:
        segment = fix_timestamp_anomalies(original)
        if not str(segment.get("text", "")).strip():
            continue
        for part in split_long_segment(segment, max_duration):
            part = {"start": part["start"], "end": part["end"],
                    "text": compress_repetitions(str(part["text"]).strip())}
            vocalization = _vocalization(part["text"])
            if (result and vocalization and _vocalization(result[-1]["text"]) == vocalization
                    and 0 <= part["start"] - result[-1]["end"] <= max_gap):
                result[-1]["end"] = max(result[-1]["end"], part["end"])
            else:
                result.append(part)
    return result
