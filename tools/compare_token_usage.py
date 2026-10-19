"""Bandingkan token dan maksud aksi sebelum/sesudah optimasi, tanpa robot.

Default --dry memeriksa fixture lokal. --run mengirim teks fixture ke Groq;
tidak ada mikrofon, TTS, atau eksekusi gerakan. Kefasihan perlu tinjauan manual.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def skill(command, seconds=None):
    return {"kind": "skill", "command": command, **({"seconds": seconds} if seconds is not None else {})}


def head(direction, *, angle=None, hold=None):
    return {"kind": "head", "direction": direction,
            **({"angle": angle} if angle is not None else {}), **({"hold": hold} if hold is not None else {})}


def _context_history():
    history = []
    for language, text in (("en", "How are you?"), ("id", "Apa kabar?"), ("fr", "Comment vas-tu?")) * 3:
        history.extend(({"role": "user", "content": f"[bahasa: {language}] {text}"},
                        {"role": "assistant", "content": '{"reply":"Bien!","actions":[]}'}))
    history.extend((
        {"role": "user", "content": "[bahasa: id] Jalan selama tiga detik, lalu duduk."},
        {"role": "assistant", "content": json.dumps({"reply": "Siap, aku berjalan tiga detik lalu duduk.", "actions": [
            {"type": "skill", "command": "kwkF", "seconds": 3},
            {"type": "skill", "command": "ksit", "seconds": 2}]}, ensure_ascii=False)},
    ))
    return tuple(history)


CASES = (
    {"id": "en-order-duration", "language": "en", "text": "Walk for three seconds, then sit down.", "expected": [skill("kwkF", 3), skill("ksit")]},
    {"id": "zh-order-duration", "language": "zh", "text": "走三秒钟，然后坐下。", "expected": [skill("kwkF", 3), skill("ksit")]},
    {"id": "hi-negation", "language": "hi", "text": "मत बैठो, सिर्फ खड़े हो जाओ।", "expected": [skill("kup")]},
    {"id": "es-order-duration", "language": "es", "text": "Camina durante tres segundos y luego siéntate.", "expected": [skill("kwkF", 3), skill("ksit")]},
    {"id": "ar-negation", "language": "ar", "text": "لا تجلس، قف فقط.", "expected": [skill("kup")]},
    {"id": "fr-order-duration", "language": "fr", "text": "Marche pendant trois secondes, puis assieds-toi.", "expected": [skill("kwkF", 3), skill("ksit")]},
    {"id": "pt-negation", "language": "pt", "text": "Não te sentes; fica de pé.", "expected": [skill("kup")]},
    {"id": "id-order-duration", "language": "id", "text": "Jalan selama tiga detik, lalu duduk.", "expected": [skill("kwkF", 3), skill("ksit")]},
    {"id": "ja-negation", "language": "ja", "text": "座らないで、立ってください。", "expected": [skill("kup")]},
    {"id": "ko-order-duration", "language": "ko", "text": "삼 초 동안 걷고 나서 앉아 주세요.", "expected": [skill("kwkF", 3), skill("ksit")]},
    {"id": "id-mixed-right", "language": "id", "text": "Duduk, tengok kanan, lalu berdiri.", "expected": [skill("ksit"), head(1), skill("kup")]},
    {"id": "fr-mixed-left", "language": "fr", "text": "Assieds-toi, regarde à gauche, puis lève-toi.", "expected": [skill("ksit"), head(-1), skill("kup")]},
    {"id": "id-repeat-skills", "language": "id", "text": "Duduk, berdiri, lalu duduk lagi.", "expected": [skill("ksit"), skill("kup"), skill("ksit")]},
    {"id": "en-repeat-head", "language": "en", "text": "Look right twice.", "expected": [head(1), head(1)]},
    {"id": "id-custom-head", "language": "id", "text": "Tengok kanan 25 derajat dan tahan 0,8 detik.", "expected": [head(1, angle=25, hold=0.8)]},
    {"id": "id-unsupported-legs", "language": "id", "text": "Buat gerakan kaki original sambil menganggukkan kepala.", "expected": []},
    {"id": "fr-chat", "language": "fr", "text": "Comment te sens-tu aujourd'hui ?", "expected": []},
    {"id": "ko-context-repeat", "language": "ko", "text": "방금 동작을 다시 해 줘.", "history": _context_history(), "expected": [skill("kwkF", 3), skill("ksit")]},
)


def _project_path(value):
    try:
        path = Path(value)
        path = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
        path.relative_to(PROJECT_ROOT.resolve())
        return path
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError("Baseline dan laporan harus berada di dalam project.") from error


def load_baseline(path=None):
    if path is None:
        candidates = sorted((PROJECT_ROOT / "logs" / "measurements").glob("*-token-optimization/baseline.json"))
        if not candidates:
            raise ValueError("Snapshot baseline token tidak ditemukan; gunakan --baseline.")
        path = candidates[-1]
    path = _project_path(path)
    try:
        baseline = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("Snapshot baseline tidak dapat dibaca sebagai JSON.") from error
    if (not isinstance(baseline, dict)
            or baseline.get("model") != "openai/gpt-oss-120b"
            or baseline.get("temperature") != 0
            or baseline.get("max_history_messages") != 20
            or any(not isinstance(baseline.get(key), str) or not baseline[key].strip()
                   for key in ("system_prompt", "motion_prompt"))
            or not isinstance(baseline.get("commands"), dict) or not baseline["commands"]
            or any(not isinstance(key, str) or not isinstance(value, str) or not value
                   for key, value in baseline["commands"].items())):
        raise ValueError("Baseline harus memuat prompt, commands, model/temperature yang sama dan riwayat 20 pesan.")
    return {**baseline, "source_path": str(path)}


def build_messages(prompt, case):
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    history = case.get("history", ())
    if len(history) > 20:
        raise ValueError("Fixture tidak boleh memotong atau melebihi riwayat aplikasi 20 pesan.")
    messages = [SystemMessage(content=prompt)]
    for message in history:
        cls = HumanMessage if message["role"] == "user" else AIMessage
        messages.append(cls(content=message["content"]))
    messages.append(HumanMessage(content=f"[bahasa: {case['language']}] {case['text']}"))
    return messages


def action_signature(actions):
    signature = []
    for action in actions:
        if action["type"] == "skill":
            signature.append(skill(action["command"], action["seconds"]))
            continue
        current = None
        for step in action["steps"]:
            angle = step["joints"].get(0, 0)
            if not angle:
                current = None
                continue
            direction = 1 if angle > 0 else -1
            if current is None or current["direction"] != direction:
                current = head(direction, angle=abs(angle), hold=step["hold"])
                signature.append(current)
            else:
                current.update(angle=max(current["angle"], abs(angle)), hold=step["hold"])
    return signature


def score_actions(case, reply, actions):
    actual = action_signature(actions)
    if not isinstance(reply, str) or not reply.strip() or len(actual) != len(case["expected"]):
        return False
    return all(
        all(key in found and (math.isclose(float(value), float(found[key]), abs_tol=1e-6)
                              if key in {"seconds", "angle", "hold"} else value == found[key])
            for key, value in expected.items())
        for expected, found in zip(case["expected"], actual)
    )


class RequestFailure(RuntimeError):
    def __init__(self, error):
        from tools.check_language_understanding import retry_after_seconds

        self.error_type = type(error).__name__
        code = getattr(error, "status_code", None)
        self.status_code = code if isinstance(code, int) and not isinstance(code, bool) else None
        self.retry_after_s = retry_after_seconds(error)
        super().__init__("Panggilan model gagal; rincian sensitif tidak disimpan.")


class APIPacer:
    """One shared delay for every invocation, including both head planners."""
    def __init__(self, model, delay=12.0):
        if not math.isfinite(delay) or delay < 0:
            raise ValueError("Jeda API harus finite dan nonnegatif.")
        self.model, self.delay, self.calls = model, delay, 0

    def invoke(self, messages):
        import llm_utils

        if self.calls:
            remaining = self.delay
            while remaining > 0:
                wait = min(remaining, 30.0)
                time.sleep(wait)
                remaining -= wait
        self.calls += 1
        started = time.perf_counter()
        try:
            return self.model.invoke(messages)
        except BaseException as error:
            llm_utils.token_counter.add(SimpleNamespace(response_metadata={}), source="failed_request",
                                        elapsed_s=time.perf_counter() - started)
            if isinstance(error, Exception):
                raise RequestFailure(error) from None
            raise


def _response_text(response):
    text = response.content or ""
    return text if text.strip() else response.additional_kwargs.get("reasoning_content", "") or ""


def _baseline_head_plan(model, request, prompt, counter):
    import motion_gen
    from langchain_core.messages import HumanMessage, SystemMessage
    from llm_utils import strip_thinking

    started = time.perf_counter()
    response = model.invoke([SystemMessage(content=prompt), HumanMessage(content=request)])
    counter.add(response, source="head_planner", elapsed_s=time.perf_counter() - started)
    plan = motion_gen.parse_plan(strip_thinking(_response_text(response)))
    steps, warnings = motion_gen.validate_plan(plan)
    steps, direction_warnings = motion_gen.enforce_direction(steps, request)
    return steps, warnings + direction_warnings


@contextmanager
def _evaluation_environment(variant, baseline, counter):
    import llm_utils
    import motion_gen

    original_counter, original_planner = llm_utils.token_counter, motion_gen.plan_from_request
    llm_utils.token_counter = counter
    if variant == "before":
        motion_gen.plan_from_request = lambda model, request: _baseline_head_plan(
            model, request, baseline["motion_prompt"], counter,
        )
    try:
        yield
    finally:
        motion_gen.plan_from_request = original_planner
        llm_utils.token_counter = original_counter


def evaluate_variant(model, case, baseline, variant, *, counter):
    import action_sequence
    from llm_utils import strip_thinking

    if variant not in {"before", "after"}:
        raise ValueError("Varian harus before/after.")
    checkpoint, started, calls = counter.snapshot(), time.perf_counter(), model.calls
    history = case.get("history", ())
    result = {"case_id": case["id"], "variant": variant, "language": case["language"],
              "text": case["text"], "history_messages": len(history),
              "history_digest": hashlib.sha256(json.dumps(history, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
              "status": "ok", "correct": False}
    with _evaluation_environment(variant, baseline, counter):
        try:
            prompt = baseline["system_prompt"] if variant == "before" else action_sequence.build_system_prompt(baseline["commands"])
            invoke_started = time.perf_counter()
            response = model.invoke(build_messages(prompt, case))
            counter.add(response, source="conversation", elapsed_s=time.perf_counter() - invoke_started)
            reply, actions = action_sequence.parse_reply(
                strip_thinking(_response_text(response)), set(baseline["commands"].values()), request_text=case["text"],
            )
            actions = action_sequence.validate_actions(actions, set(baseline["commands"].values()))
            prepared, warnings = action_sequence.prepare_actions(model, actions, set(baseline["commands"].values()))
            result.update(reply=reply, actions=prepared, warnings=warnings,
                          signature=action_signature(prepared), correct=score_actions(case, reply, prepared))
        except action_sequence.ActionPlanError as error:
            result.update(status="invalid_plan", error_type=type(error).__name__)
        except RequestFailure as error:
            result.update(status="llm_error", error_type=error.error_type,
                          http_status=error.status_code, retry_after_s=error.retry_after_s)
        except Exception as error:
            result.update(status="error", error_type=type(error).__name__)
    result.update(elapsed_s=time.perf_counter() - started, attempted_calls=model.calls - calls,
                  tokens=counter.delta(checkpoint), token_records=counter.records_since(checkpoint))
    return result


def summarize(records, counters):
    paired, incomplete = [], []
    for case_id in dict.fromkeys(record["case_id"] for record in records):
        pair = {record["variant"]: record for record in records if record["case_id"] == case_id}
        if set(pair) == {"before", "after"}:
            paired.append(pair)
        if set(pair) != {"before", "after"} or any(record["status"] != "ok" for record in pair.values()):
            incomplete.append({"case_id": case_id,
                               "variants": {variant: record["status"] for variant, record in pair.items()}})
    measured = [pair for pair in paired if all(record["tokens"]["total_tokens"] is not None for record in pair.values())]
    scored = [pair for pair in paired if all(record["status"] in {"ok", "invalid_plan"} for record in pair.values())]
    before = sum(pair["before"]["tokens"]["total_tokens"] for pair in measured)
    after = sum(pair["after"]["tokens"]["total_tokens"] for pair in measured)
    return {
        "paired_cases": len(paired), "token_measured_pairs": len(measured), "semantic_scored_pairs": len(scored),
        "incomplete_or_error_pairs": incomplete,
        "paired_before_tokens": before if measured else None, "paired_after_tokens": after if measured else None,
        "token_savings_percent": (before - after) / before * 100 if before else None,
        "semantic_regressions": [pair["before"]["case_id"] for pair in scored if pair["before"]["correct"] and not pair["after"]["correct"]],
        "semantic_improvements": [pair["before"]["case_id"] for pair in scored if not pair["before"]["correct"] and pair["after"]["correct"]],
        "variants": {variant: {"tested": sum(record["variant"] == variant for record in records),
                               "passed": sum(record["variant"] == variant and record["correct"] for record in records),
                               "tokens": counter.summary()} for variant, counter in counters.items()},
    }


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry", action="store_true", help="periksa tanpa API (default)")
    mode.add_argument("--run", action="store_true", help="kirim hanya fixture teks ke Groq")
    parser.add_argument("--baseline", help="snapshot baseline di dalam project; default snapshot terbaru")
    parser.add_argument("--max-cases", type=int, choices=range(1, len(CASES) + 1), default=len(CASES))
    parser.add_argument("--delay", type=float, default=12.0, help="jeda setiap panggilan, termasuk kepala")
    args = parser.parse_args(argv)
    if not math.isfinite(args.delay) or args.delay < 0:
        parser.error("--delay harus finite dan nonnegatif")
    try:
        baseline = load_baseline(args.baseline)
    except ValueError as error:
        parser.error(str(error))
    import action_sequence
    import motion_gen
    from llm_utils import TokenCounter

    cases = CASES[:args.max_cases]
    after_prompt = action_sequence.build_system_prompt(baseline["commands"])
    after_snapshot = {"system_prompt": after_prompt, "motion_prompt": motion_gen.build_motion_prompt(),
                      "system_prompt_chars": len(after_prompt),
                      "system_prompt_sha256": hashlib.sha256(after_prompt.encode()).hexdigest()}
    for case in cases:
        build_messages(baseline["system_prompt"], case)
    print("[Uji] Pasangan fixture teks yang sama; tanpa robot, mikrofon, TTS, atau perubahan model.")
    print("[Uji] Menilai maksud/urutan/durasi/jumlah/arah kepala; kefasihan tidak diberi skor otomatis.")
    if not args.run:
        for case in cases:
            print(f"{case['id']}: {case['text']} (riwayat {len(case.get('history', ()))} pesan)")
        print(f"[Dry] {len(cases)} pasangan siap; tidak memanggil API.")
        return 0

    stamp = datetime.now(timezone(timedelta(hours=7))).strftime("%Y%m%d-%H%M%S-%f")
    directory = _project_path(PROJECT_ROOT / "logs" / "measurements" / f"{stamp}-token-comparison")
    directory.mkdir(parents=True)
    counters = {variant: TokenCounter() for variant in ("before", "after")}
    records, status, error_type = [], "complete", None
    try:
        from dotenv import load_dotenv
        from llm_utils import get_model

        load_dotenv(PROJECT_ROOT / ".env")
        model = APIPacer(get_model("groq"), args.delay)
        stop = False
        for case in cases:
            for variant in ("before", "after"):
                record = evaluate_variant(model, case, baseline, variant, counter=counters[variant])
                records.append(record)
                with (directory / "records.jsonl").open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                print(f"{case['id']} {variant}: {'sesuai' if record['correct'] else 'belum sesuai'}; token {record['tokens']['total_tokens']}; {record['attempted_calls']} panggilan")
                if record["status"] == "llm_error":
                    print(f"[API] Berhenti setelah permintaan gagal; HTTP {record.get('http_status')}; Retry-After: {record.get('retry_after_s')} detik.")
                    status, stop = ("rate_limited" if record.get("http_status") == 429 else "llm_error"), True
                    break
            if stop:
                break
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as error:
        status, error_type = "error", type(error).__name__
    finally:
        report = {"kind": "paired_token_capability_comparison", "status": status,
                  "baseline": baseline["source_path"], "model": baseline["model"], "temperature": baseline["temperature"],
                  "max_history_messages": 20, "planned_cases": len(cases), "delay_per_call_s": args.delay,
                  "after_snapshot": after_snapshot,
                  "timing_caveat": "Includes comparison pacing; not production latency.",
                  "unscored": ["reply fluency", "pronunciation", "hardware execution"],
                  "cases": cases, "records": records, **summarize(records, counters)}
        if error_type:
            report["error_type"] = error_type
        path = directory / "summary.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"[Laporan] {path}")
    failed = (status != "complete" or any(record["status"] != "ok" for record in records)
              or any(record["variant"] == "after" and not record["correct"] for record in records))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
