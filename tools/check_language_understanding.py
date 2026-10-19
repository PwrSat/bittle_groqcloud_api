"""Uji pemahaman teks 10 bahasa tanpa robot, mikrofon, atau TTS.

Default --dry hanya memeriksa contoh lokal. --run memakai kuota Groq.
--session latest memakai transkripsi asli dari sesi rekaman terpandu terbaru.
Skor mencakup aksi, negasi, urutan dan durasi; bukan kefasihan atau akurasi suara.
"""

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LANGUAGES = ("en", "zh", "hi", "es", "ar", "fr", "pt", "id", "ja", "ko")
PHRASES = {
    "en": ("Sit down.", "Don't sit down; stand up only.",
           "Walk for three seconds, then sit down."),
    "zh": ("坐下。", "不要坐下，只要站起来。", "走三秒钟，然后坐下。"),
    "hi": ("बैठो।", "मत बैठो, सिर्फ खड़े हो जाओ।", "तीन सेकंड चलो, फिर बैठो।"),
    "es": ("Siéntate.", "No te sientes; solo ponte de pie.",
           "Camina durante tres segundos y luego siéntate."),
    "ar": ("اجلس.", "لا تجلس، قف فقط.", "امشِ لمدة ثلاث ثوانٍ، ثم اجلس."),
    "fr": ("Assieds-toi.", "Ne t'assieds pas ; reste debout.",
           "Marche pendant trois secondes, puis assieds-toi."),
    "pt": ("Senta-te.", "Não te sentes; fica de pé.",
           "Anda durante três segundos e depois senta-te."),
    "id": ("Duduk.", "Jangan duduk, berdiri saja.",
           "Jalan selama tiga detik, lalu duduk."),
    "ja": ("座って。", "座らないで、立って。", "3秒歩いてから座って。"),
    "ko": ("앉아.", "앉지 말고 서 있어.", "3초 동안 걷고 나서 앉아."),
}
CASES = tuple(
    {"id": f"{language}-{kind}", "language": language, "text": text,
     "expected": list(expected), **({"first_seconds": 3.0} if index == 2 else {})}
    for language in LANGUAGES
    for index, (kind, text, expected) in enumerate(zip(
        ("sit", "negation", "order-duration"), PHRASES[language],
        (("ksit",), ("kup",), ("kwkF", "ksit")),
    ))
)


def resolve_session(value):
    """Find a saved capture inside this project, including resolved symlinks."""
    root = PROJECT_ROOT.resolve()
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("--session harus berupa folder, utterances.jsonl, atau latest")
    if str(value) == "latest":
        matches = sorted((root / "logs" / "measurements").glob("*-multilingual/utterances.jsonl"))
        if not matches:
            raise ValueError("Belum ada sesi multibahasa. Jalankan tools/check_multilingual.py --guided dahulu.")
        candidate = matches[-1]
    else:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.name != "utterances.jsonl":
            candidate = candidate / "utterances.jsonl"
    try:
        candidate = candidate.resolve()
        candidate.relative_to(root)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError("Sesi harus berada di dalam folder project ini.") from error
    if not candidate.is_file():
        raise ValueError("Berkas utterances.jsonl tidak ditemukan. Pilih folder sesi rekaman yang sudah selesai.")
    return candidate


def load_session_cases(value):
    """Use actual ASR text/tag; guided references describe scoring only."""
    from tools.check_multilingual import REFERENCE_CASES

    source = resolve_session(value)
    references = dict(REFERENCE_CASES)
    cases, seen = [], set()
    try:
        with source.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                if len(cases) >= 30:
                    raise ValueError("Sesi berisi lebih dari 30 ucapan; pilih sesi terpandu maksimal 30 ucapan.")
                try:
                    row = json.loads(line)
                except (ValueError, TypeError) as error:
                    raise ValueError(f"JSON sesi rusak pada baris {number}; pilih berkas rekaman yang lengkap.") from error
                if not isinstance(row, dict):
                    raise ValueError(f"Baris {number}: data ucapan harus berupa objek JSON.")
                sample = row.get("sample")
                if isinstance(sample, bool) or not isinstance(sample, int) or sample <= 0 or sample in seen:
                    raise ValueError(f"Baris {number}: sample harus bilangan bulat positif yang tidak berulang.")
                language = row.get("reference_language")
                if (not isinstance(language, str) or language not in references
                        or row.get("reference") != references[language]):
                    raise ValueError(f"Baris {number}: acuan terpandu hilang/tidak cocok; rekam dengan --guided.")
                speech = row.get("speech")
                if not isinstance(speech, dict) or speech.get("status") != "ok":
                    raise ValueError(f"Baris {number}: rekaman belum berhasil; pilih sesi dengan status suara ok.")
                text, detected = speech.get("normalized_text"), speech.get("language_returned")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"Baris {number}: transkripsi normalized_text kosong; rekam ulang ucapan ini.")
                if not isinstance(detected, str) or detected not in LANGUAGES:
                    raise ValueError(f"Baris {number}: bahasa hasil transkripsi hilang/tidak didukung.")
                cases.append({"id": f"recorded-{sample:02d}-{language}", "language": detected,
                              "text": text, "reference_language": language,
                              "reference": references[language], "source_session": str(source),
                              "expected": ["kwkF", "ksit"], "first_seconds": 3.0})
                seen.add(sample)
    except (OSError, UnicodeError) as error:
        raise ValueError("Sesi tidak dapat dibaca sebagai UTF-8; pilih utterances.jsonl yang lengkap.") from error
    if not cases:
        raise ValueError("Sesi belum berisi ucapan. Rekam dengan tools/check_multilingual.py --guided dahulu.")
    return cases


def score_actions(case, actions):
    """Exact ordered skills; a forbidden sit or unexpected head action fails."""
    if any(action.get("type") != "skill" for action in actions):
        return False
    if [action.get("command") for action in actions] != case["expected"]:
        return False
    return "first_seconds" not in case or actions[0].get("seconds") == case["first_seconds"]


def build_messages(prompt, case):
    from langchain_core.messages import HumanMessage, SystemMessage

    return [SystemMessage(content=prompt),
            HumanMessage(content=f"[bahasa: {case['language']}] {case['text']}")]


def retry_after_seconds(error):
    """Read a numeric provider hint, without saving arbitrary error text."""
    headers = getattr(getattr(error, "response", None), "headers", {}) or {}
    try:
        value = float(headers.get("retry-after", "nan"))
        return value if math.isfinite(value) and value >= 0 else None
    except (TypeError, ValueError, AttributeError):
        return None


def evaluate_case(model, case, commands, counter):
    import action_sequence
    from llm_utils import strip_thinking

    checkpoint = counter.snapshot()
    started = time.perf_counter()
    result = {**case, "status": "ok", "correct": False}
    try:
        response = model.invoke(build_messages(action_sequence.build_system_prompt(commands), case))
        counter.add(response, source="language_understanding",
                    elapsed_s=time.perf_counter() - started)
        raw = response.content or response.additional_kwargs.get("reasoning_content", "")
        reply, actions = action_sequence.parse_reply(
            strip_thinking(raw), set(commands.values()), request_text=case["text"],
        )
        # These fixtures need no head planner: validate without ever generating motion.
        actions = action_sequence.validate_actions(actions, set(commands.values()))
        result.update(reply=reply, actions=actions, correct=score_actions(case, actions))
    except action_sequence.ActionPlanError as error:
        result.update(status="invalid", error_type=type(error).__name__)
    except Exception as error:
        status_code = getattr(error, "status_code", None)
        result.update(status="llm_error", error_type=type(error).__name__,
                      http_status=status_code if isinstance(status_code, int) else None,
                      retry_after_s=retry_after_seconds(error))
    result.update(elapsed_s=time.perf_counter() - started, tokens=counter.delta(checkpoint))
    if result["status"] == "llm_error":
        result["tokens"].update(usage_known=False, total_tokens=None,
                                prompt_tokens=None, completion_tokens=None, cached_tokens=None)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry", action="store_true", help="periksa contoh, tanpa API (default)")
    mode.add_argument("--run", action="store_true", help="panggil Groq; memakai kuota API")
    parser.add_argument("--session", help="folder/utterances.jsonl/latest; uji transkripsi rekaman asli")
    parser.add_argument("--languages", default=",".join(LANGUAGES), help="kode dipisah koma")
    parser.add_argument("--max-cases", type=int, choices=range(1, 31))
    parser.add_argument("--delay", type=float, default=12.0, help="jeda antarpanggilan, default 12 detik")
    args = parser.parse_args(argv)
    languages = set(args.languages.split(","))
    if not languages or not languages <= set(LANGUAGES):
        parser.error("--languages harus berisi kode: " + ",".join(LANGUAGES))
    if not math.isfinite(args.delay) or args.delay < 0:
        parser.error("--delay harus angka finite nonnegatif")
    try:
        available = load_session_cases(args.session) if args.session else CASES
    except ValueError as error:
        parser.error(str(error))
    cases = [case for case in available
             if case.get("reference_language", case["language"]) in languages][:args.max_cases]
    if not cases:
        parser.error("Tidak ada ucapan pada sesi ini untuk --languages yang dipilih.")
    source_session = cases[0]["source_session"] if args.session else None
    commands = json.loads((PROJECT_ROOT / "Commands.json").read_text(encoding="utf-8"))
    import action_sequence

    for case in cases:
        expected = [{"type": "skill", "command": command,
                     "seconds": case.get("first_seconds", 2) if index == 0 else 2}
                    for index, command in enumerate(case["expected"])]
        action_sequence.validate_actions(expected, set(commands.values()))
    action_sequence.build_system_prompt(commands)
    print("[Uji] Teks saja; tidak ada robot, rekaman, atau pemutaran suara.")
    print("[Uji] Arab standar, Mandarin tertulis, Hindi Devanagari, Korea Hangul.")
    if args.session:
        print(f"[Uji] Transkripsi asli dari {source_session}")
        print("[Skor] Sesuai maksud acuan: jalan tiga detik, lalu duduk. Kesalahan ASR ikut memengaruhi hasil.")
        print("[Skor] Kesesuaian terhadap maksud acuan berbeda dari kepatuhan AI terhadap teks ASR yang salah.")
    if not args.run:
        for case in cases:
            tag = f" [bahasa: {case['language']}]" if args.session else ""
            print(f"{case['id']}{tag}: {case['text']} -> {case['expected']}")
        print(f"[Dry] {len(cases)} contoh valid; tidak memanggil API. Gunakan --run untuk mengukur Groq.")
        return 0

    from dotenv import load_dotenv
    from llm_utils import TokenCounter, get_model

    load_dotenv(PROJECT_ROOT / ".env")
    model, counter, records = get_model("groq"), TokenCounter(), []
    stamp = datetime.now(timezone(timedelta(hours=7))).strftime("%Y%m%d-%H%M%S-%f")
    directory = PROJECT_ROOT / "logs" / "measurements" / f"{stamp}-language-understanding"
    directory.mkdir(parents=True)
    try:
        for index, case in enumerate(cases):
            if index:
                print(f"[Kuota] Jeda {args.delay:g} detik sebelum contoh berikutnya.")
                remaining = args.delay
                while remaining > 0:
                    time.sleep(min(remaining, 30))
                    remaining -= min(remaining, 30)
            record = evaluate_case(model, case, commands, counter)
            records.append(record)
            print(f"{case['id']}: {'sesuai' if record['correct'] else 'belum sesuai'} ({record['status']})")
            if args.session:
                print("[AI] " + json.dumps({key: record.get(key) for key in ("reply", "actions")}, ensure_ascii=False))
            if record.get("http_status") == 429:
                print(f"[Kuota] Batas API; berhenti. Retry-After: {record.get('retry_after_s')} detik.")
                break
    except KeyboardInterrupt:
        print("\n[Uji] Dihentikan; hasil yang selesai tetap disimpan.")
    finally:
        passed = sum(record["correct"] for record in records)
        summary = {"kind": "text_understanding",
                   "source": "recorded_transcripts" if args.session else "reference_text",
                   "scope": "recorded_transcript_to_intended_actions" if args.session else "ordered_skills_negation_duration",
                   "planned": len(cases), "tested": len(records), "passed": passed,
                   "pass_rate": passed / len(records) if records else None,
                   "failed_requests": sum(record["status"] == "llm_error" for record in records),
                   "tokens": counter.summary(), "records": records}
        if source_session is not None:
            summary["source_session"] = source_session
        if any(record["status"] == "llm_error" for record in records):
            summary["tokens"].update(usage_known=False, total_tokens=None,
                                     prompt_tokens=None, completion_tokens=None, cached_tokens=None)
        report = directory / "summary.json"
        report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[Hasil] {passed}/{len(records)} contoh sesuai; laporan: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
