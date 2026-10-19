import json
from pathlib import Path
import tempfile
import unittest

from baseline_metrics import percentile, score_transcript, summarize_records, write_json, write_jsonl


def utterance(reference, raw, normalized=None, **changes):
    record = {
        "kind": "utterance", "case_id": "sit", "condition": "quiet",
        "reference": reference,
        "speech": {"status": "ok", "capture_s": 2, "audio_s": 1, "transcribe_s": 0.5},
        "accuracy": {
            "raw": score_transcript(reference, raw),
            "normalized": score_transcript(reference, raw if normalized is None else normalized),
        },
        "tokens": {"calls": 1, "prompt_tokens": 100, "completion_tokens": 20,
                   "total_tokens": 120, "cached_tokens": 0, "usage_known": True},
        "llm_s": 1.5, "plan_status": "ok", "plan_correct": None,
    }
    record.update(changes)
    return record


class TranscriptScoreTests(unittest.TestCase):
    def test_unicode_case_and_punctuation_are_normalized(self):
        score = score_transcript("ＤＵＤＵＫ, lalu ‘TENGOK-KANAN’!", "duduk lalu tengok kanan")
        self.assertEqual(score, {"reference_words": 4, "word_errors": 0,
                                 "wer": 0, "exact_match": True})

    def test_substitution_insertion_and_deletion(self):
        self.assertEqual(score_transcript("duduk lalu berdiri", "duduk lalu tidur")["word_errors"], 1)
        self.assertEqual(score_transcript("duduk", "duduk lalu berdiri")["wer"], 2)
        self.assertEqual(score_transcript("duduk lalu berdiri", "duduk")["word_errors"], 2)

    def test_missing_hypothesis_counts_as_all_words_deleted(self):
        for hypothesis in (None, "", "...!"):
            with self.subTest(hypothesis=hypothesis):
                score = score_transcript("duduk lalu berdiri", hypothesis)
                self.assertEqual(score["wer"], 1)
                self.assertEqual(score["word_errors"], 3)
                self.assertFalse(score["exact_match"])

    def test_empty_reference_has_no_wer(self):
        self.assertIsNone(score_transcript("", "duduk")["wer"])
        self.assertTrue(score_transcript("...", None)["exact_match"])


class PercentileTests(unittest.TestCase):
    def test_nearest_rank_and_missing_values(self):
        values = [None, "unknown", float("nan"), float("inf"), True, 9, 1, 3, 5]
        self.assertEqual(percentile(values, 50), 3)
        self.assertEqual(percentile(values, 95), 9)
        self.assertEqual(percentile(values, 0), 1)
        self.assertEqual(percentile(values, 100), 9)
        self.assertIsNone(percentile([None, float("nan")], 95))

    def test_out_of_range_percentile_rejected(self):
        for p in (-1, 101, float("nan"), None):
            with self.subTest(p=p), self.assertRaises(ValueError):
                percentile([1, 2], p)


class SummaryTests(unittest.TestCase):
    def test_weighted_wer_includes_failed_speech_and_separates_normalization(self):
        records = [
            {"kind": "session", "tokens": {"calls": 100}},
            utterance("duduk", "duh duk", "duduk", plan_correct=True),
            utterance("duduk lalu tengok kanan", "",
                      speech={"status": "empty"}, tokens={"calls": 0, "usage_known": True},
                      llm_s=None, plan_status="no_speech", plan_correct=False),
        ]
        summary = summarize_records(records)
        self.assertEqual(summary["trials"], 2)
        self.assertEqual(summary["speech_status_counts"], {"ok": 1, "empty": 1})
        self.assertEqual(summary["accuracy"]["raw"]["wer"], 6 / 5)
        self.assertEqual(summary["accuracy"]["normalized"]["wer"], 4 / 5)
        self.assertEqual(summary["accuracy"]["normalized"]["exact_count"], 1)
        self.assertEqual(summary["tokens"]["total_tokens"], 120)
        self.assertEqual(summary["plan_checked_count"], 2)
        self.assertEqual(summary["plan_correct_count"], 1)

    def test_timing_uses_true_median_and_reports_unknown(self):
        records = [utterance("duduk", "duduk", llm_s=value,
                             speech={"status": "ok", "transcribe_s": value})
                   for value in (1, 3, None, float("nan"), -1)]
        timing = summarize_records(records)["timing"]
        self.assertEqual(timing["llm_s"], {"count": 2, "median": 2, "p95": 3})
        self.assertEqual(timing["transcribe_s"], timing["llm_s"])
        self.assertEqual(timing["capture_s"], {"count": 0, "median": None, "p95": None})

    def test_unknown_usage_is_not_reported_as_zero(self):
        records = [utterance("duduk", "duduk"),
                   utterance("duduk", "duduk", tokens={"calls": 2, "usage_known": False})]
        tokens = summarize_records(records)["tokens"]
        self.assertEqual(tokens["calls"], 3)
        self.assertFalse(tokens["usage_known"])
        for name in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens"):
            self.assertIsNone(tokens[name])

    def test_unknown_cache_does_not_invalidate_other_usage(self):
        record = utterance("duduk", "duduk")
        record["tokens"]["cached_tokens"] = None
        tokens = summarize_records([record])["tokens"]
        self.assertIsNone(tokens["cached_tokens"])
        self.assertEqual(tokens["total_tokens"], 120)
        self.assertTrue(tokens["usage_known"])

    def test_failed_request_without_response_still_makes_usage_unknown(self):
        record = utterance("duduk", "duduk", failed_requests=1,
                           tokens={"calls": 0, "usage_known": False},
                           plan_status="llm_error")
        tokens = summarize_records([record])["tokens"]
        self.assertEqual(tokens["calls"], 0)
        self.assertEqual(tokens["failed_requests"], 1)
        self.assertFalse(tokens["usage_known"])
        self.assertIsNone(tokens["total_tokens"])

    def test_empty_summary_preserves_absence_of_accuracy_and_timing(self):
        summary = summarize_records([])
        self.assertEqual(summary["trials"], 0)
        self.assertIsNone(summary["accuracy"]["raw"]["wer"])
        self.assertEqual(summary["tokens"]["calls"], 0)
        self.assertEqual(summary["tokens"]["total_tokens"], 0)
        self.assertEqual(summary["plan_checked_count"], 0)


class JsonWriterTests(unittest.TestCase):
    def test_append_records_and_write_summary_as_utf8(self):
        with tempfile.TemporaryDirectory() as folder:
            records = Path(folder) / "measurements" / "speech.jsonl"
            write_jsonl(records, {"text": "tengok kanan — siap"})
            write_jsonl(records, {"text": "duduk"})
            lines = records.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["text"], "tengok kanan — siap")
            self.assertIn("—", lines[0])
            summary = records.with_suffix(".json")
            write_json(summary, {"trials": 2})
            self.assertEqual(json.loads(summary.read_text(encoding="utf-8")), {"trials": 2})


if __name__ == "__main__":
    unittest.main()
