import unittest
from unittest.mock import Mock

from app.pipeline import (
    LLMTranslationProvider,
    MAX_PARTIAL_RETRIES,
    PipelineError,
    build_chunks,
    build_srt_content,
    parse_chunk_output,
)


class TranslationValidationTests(unittest.TestCase):
    def test_rejects_missing_duplicate_reordered_empty_and_unnumbered(self):
        for response in ("0|甲\n2|丙", "0|甲\n0|乙\n2|丙", "1|乙\n0|甲\n2|丙",
                         "0|甲\n1|\n2|丙", "甲\n乙\n丙", "0|甲\n1|乙\n2|丙\n3|额外"):
            with self.subTest(response=response), self.assertRaises(PipelineError):
                parse_chunk_output(response, [0, 1, 2], ["a", "b", "c"])

    def test_retries_missing_suffix_without_shifting_translations(self):
        provider = LLMTranslationProvider.__new__(LLMTranslationProvider)
        provider._request_translation = Mock(side_effect=["0|甲\n2|丙", "1|乙\n2|丙"])
        chunk = build_chunks([{"text": value} for value in ("a", "b", "c")])[0]
        self.assertEqual(provider.translate_chunk(chunk, "zh", "general", ""), ["甲", "乙", "丙"])
        retried = provider._request_translation.call_args_list[1].args[1]
        self.assertIn("1|b", retried)
        self.assertIn("2|c", retried)

    def test_exhaustion_uses_original_only_for_unverified_suffix(self):
        provider = LLMTranslationProvider.__new__(LLMTranslationProvider)
        provider._request_translation = Mock(side_effect=["0|甲\n1|"] + ["invalid"] * MAX_PARTIAL_RETRIES)
        chunk = build_chunks([{"text": value} for value in ("a", "b", "c")])[0]
        with self.assertLogs("app.pipeline", level="WARNING") as logs:
            result = provider.translate_chunk(chunk, "zh", "general", "")
        self.assertEqual(result, ["甲", "b", "c"])
        self.assertTrue(any("回退" in line for line in logs.output))


class SubtitleDurationTests(unittest.TestCase):
    def test_extends_short_subtitles_without_new_overlap(self):
        segments = [
            {"start": 1.0, "end": 1.1, "text": "a"},
            {"start": 1.5, "end": 1.6, "text": "b"},
            {"start": 3.0, "end": 3.1, "text": "c"},
        ]
        srt = build_srt_content(segments, None)
        self.assertIn("00:00:01,000 --> 00:00:01,500", srt)
        self.assertIn("00:00:01,500 --> 00:00:02,300", srt)
        self.assertIn("00:00:03,000 --> 00:00:03,800", srt)
        self.assertEqual(segments[0]["end"], 1.1)

    def test_preserves_existing_overlaps_and_long_subtitles(self):
        segments = [
            {"start": 0.0, "end": 0.4, "text": "a"},
            {"start": 0.2, "end": 2.0, "text": "b"},
        ]
        srt = build_srt_content(segments, None)
        self.assertIn("00:00:00,000 --> 00:00:00,400", srt)
        self.assertIn("00:00:00,200 --> 00:00:02,000", srt)
