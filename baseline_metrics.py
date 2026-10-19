"""Offline measurements for voice baselines, without audio or robot imports.

WER and exact matches measure transcription text, not whether a robot action
captures the speaker's intent. Semantic plan checks are reported separately.
"""

from collections import Counter
from collections.abc import Mapping
import json
import math
from numbers import Real
from pathlib import Path
from statistics import median
import unicodedata


def _words(text):
    text = unicodedata.normalize("NFKC", text or "").casefold()
    text = "".join(
        " " if unicodedata.category(char).startswith("P") else char
        for char in text
    )
    return text.split()


def score_transcript(reference, hypothesis):
    """Return normalized word-level edit distance; missing speech is deletions.

    Punctuation, letter case and Unicode compatibility differences are ignored.
    WER is a ratio (0.25 means 25%); it is None for an empty reference and can
    exceed 1.0 when the hypothesis contains many inserted words.
    """
    reference_words = _words(reference)
    hypothesis_words = _words(hypothesis)
    previous = list(range(len(hypothesis_words) + 1))
    for row, expected in enumerate(reference_words, start=1):
        current = [row]
        for column, actual in enumerate(hypothesis_words, start=1):
            current.append(min(
                previous[column] + 1,
                current[column - 1] + 1,
                previous[column - 1] + (expected != actual),
            ))
        previous = current
    errors = previous[-1]
    word_count = len(reference_words)
    return {
        "reference_words": word_count,
        "word_errors": errors,
        "wer": errors / word_count if word_count else None,
        "exact_match": reference_words == hypothesis_words,
    }


def _finite_values(values):
    return [
        float(value) for value in values
        if isinstance(value, Real) and not isinstance(value, bool)
        and math.isfinite(value)
    ]


def percentile(values, p):
    """Nearest-rank percentile, ignoring missing/non-numeric/non-finite values."""
    if not isinstance(p, Real) or not math.isfinite(p) or not 0 <= p <= 100:
        raise ValueError("percentile must be between 0 and 100")
    valid = sorted(_finite_values(values))
    if not valid:
        return None
    return valid[max(0, math.ceil(p / 100 * len(valid)) - 1)]


def _integer(value):
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _accuracy_summary(records, stage):
    result = {"scored_count": 0, "reference_words": 0, "word_errors": 0,
              "wer": None, "exact_count": 0}
    for record in records:
        score = record.get("accuracy", {}).get(stage)
        if not isinstance(score, Mapping):
            continue
        words = _integer(score.get("reference_words"))
        errors = _integer(score.get("word_errors"))
        if words is None or errors is None:
            continue
        result["scored_count"] += 1
        result["reference_words"] += words
        result["word_errors"] += errors
        result["exact_count"] += score.get("exact_match") is True
    if result["reference_words"]:
        result["wer"] = result["word_errors"] / result["reference_words"]
    return result


def _timing_summary(values):
    valid = [value for value in _finite_values(values) if value >= 0]
    return {
        "count": len(valid),
        "median": median(valid) if valid else None,
        "p95": percentile(valid, 95),
    }


def _token_summary(records):
    tokens = [record.get("tokens", {}) for record in records]
    contributing = [item for item in tokens
                    if (_integer(item.get("calls")) or 0) > 0
                    or item.get("usage_known") is not True]
    result = {"calls": sum(_integer(item.get("calls")) or 0 for item in tokens)}
    result["failed_requests"] = sum(
        _integer(record.get("failed_requests", record.get("tokens", {}).get("failed_requests"))) or 0
        for record in records
    )
    for name in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens"):
        values = [_integer(item.get(name)) for item in contributing]
        known = all(value is not None for value in values)
        if name != "cached_tokens":
            known = known and all(item.get("usage_known") is True for item in contributing)
        result[name] = sum(values) if known else None
    result["usage_known"] = all(item.get("usage_known") is True for item in contributing)
    return result


def summarize_records(records):
    """Aggregate utterances; failed/empty speech still counts in textual WER.

    WER is weighted by reference-word count, not averaged over utterances.
    Missing token usage stays unknown instead of silently becoming zero.
    """
    utterances = [record for record in records
                  if isinstance(record, Mapping) and record.get("kind") == "utterance"]
    speech = [record.get("speech", {}) for record in utterances]
    timings = {
        name: _timing_summary(item.get(name) for item in speech)
        for name in ("capture_s", "audio_s", "transcribe_s")
    }
    timings["llm_s"] = _timing_summary(record.get("llm_s") for record in utterances)
    checked = [record.get("plan_correct") for record in utterances
               if isinstance(record.get("plan_correct"), bool)]
    return {
        "trials": len(utterances),
        "speech_status_counts": dict(Counter(item.get("status", "unknown") for item in speech)),
        "accuracy": {stage: _accuracy_summary(utterances, stage)
                     for stage in ("raw", "normalized")},
        "timing": timings,
        "tokens": _token_summary(utterances),
        "plan_checked_count": len(checked),
        "plan_correct_count": sum(checked),
    }


def write_jsonl(path, record):
    """Append one UTF-8 JSON record; propagate file errors to the caller."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def write_json(path, data):
    """Write a readable UTF-8 summary; propagate file errors to the caller."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
