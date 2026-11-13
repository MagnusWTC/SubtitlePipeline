import unittest

from app.segment_cleaner import clean_segments


class TestSegmentCleaner(unittest.TestCase):
    def test_repetition_keeps_dialogue_and_time(self):
        segments = [{"start": 325.665, "end": 353.401,
                     "text": "え?なになになになになに?何してるんですか?"},
                    {"start": 358.0, "end": 360.0, "text": "ああああああ"}]
        result = clean_segments(segments)
        self.assertEqual(result[0]["text"], "え?なに?何してるんですか?")
        self.assertEqual(result[0]["start"], 325.67)
        self.assertEqual(result[0]["end"], 353.4)
        self.assertEqual(result[1]["text"], "あー")
        self.assertEqual(result[1]["end"], 360.0)

    def test_long_untimed_text_is_not_split(self):
        segment = {"start": 448.838, "end": 466.275, "text": "おいしい"}
        result = clean_segments([segment])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "おいしい")
        self.assertEqual(result[0]["end"], 466.27)

    def test_word_timestamps_split_dialogue_at_actual_boundaries(self):
        segment = {"start": 1.0, "end": 6.0, "text": "こんにちは。 お元気ですか？", "words": [
            {"word": "こんにちは。", "start": 1.2, "end": 2.0},
            {"word": "お元気ですか？", "start": 4.0, "end": 5.0},
        ]}
        self.assertEqual(clean_segments([segment]), [
            {"start": 1.2, "end": 2.0, "text": "こんにちは。"},
            {"start": 4.0, "end": 5.0, "text": "お元気ですか？"},
        ])

    def test_incomplete_word_timing_falls_back_without_losing_text(self):
        segment = {"start": 0, "end": 20, "text": "こんにちは。さようなら。", "words": [
            {"word": "こんにちは。", "start": 1, "end": 2},
            {"word": "さようなら。"},
        ]}
        self.assertEqual(clean_segments([segment]), [{"start": 0.0, "end": 20.0, "text": segment["text"]}])

    def test_only_adjacent_matching_vocalizations_merge(self):
        result = clean_segments([
            {"start": 0, "end": 1, "text": "ああああ"},
            {"start": 1.2, "end": 2, "text": "ああああ"},
            {"start": 5, "end": 6, "text": "ああああ"},
            {"start": 6.1, "end": 7, "text": "待って"},
            {"start": 7.1, "end": 8, "text": "待って"},
        ])
        self.assertEqual(result, [
            {"start": 0.0, "end": 2.0, "text": "あー"},
            {"start": 5.0, "end": 6.0, "text": "あー"},
            {"start": 6.1, "end": 7.0, "text": "待って"},
            {"start": 7.1, "end": 8.0, "text": "待って"},
        ])


if __name__ == "__main__":
    unittest.main()
