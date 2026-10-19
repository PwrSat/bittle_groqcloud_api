"""Ukur suara dan token tanpa menghubungkan atau menggerakkan robot.

python tools/measure_baseline.py --condition tenang --with-llm
python tools/measure_baseline.py --condition ramai --with-llm
Tanpa --with-llm, pengujian hanya mengukur suara dan tidak memanggil Groq.
"""

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import action_sequence
from baseline_metrics import score_transcript, summarize_records, write_json, write_jsonl
import llm_utils
import speechtotext


CASES = (
    {"id": "01", "text": "Duduk.", "expected": ["ksit"]},
    {"id": "02", "text": "Berdiri.", "expected": ["kup"]},
    {"id": "03", "text": "Tengok kanan.", "expected": ["head_right"]},
    {"id": "04", "text": "Tengok kiri.", "expected": ["head_left"]},
    {"id": "05", "text": "Duduk, tengok kanan, lalu berdiri.",
     "expected": ["ksit", "head_right", "kup"]},
    {"id": "06", "text": "Jalan selama tiga detik, lalu duduk.",
     "expected": ["kwkF", "ksit"], "first_seconds": 3.0},
    {"id": "07", "text": "Jangan duduk, berdiri saja.", "expected": ["kup"]},
    {"id": "08", "text": "Terima kasih.", "expected": []},
)


def build_chain(model, commands):
    """Same prompt/history policy as the application; never import its entrypoint."""
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables.history import RunnableWithMessageHistory

    history = InMemoryChatMessageHistory()

    def get_history(session_id):
        if len(history.messages) > 20:
            history.messages = history.messages[-20:]
        return history

    system = action_sequence.build_system_prompt(commands)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system.replace("{", "{{").replace("}", "}}")),
        MessagesPlaceholder(variable_name="messages"),
    ])
    return RunnableWithMessageHistory(prompt | model, get_history)


def plan_matches(case, actions):
    """Check the requested order, direction, duration and negation; not physical safety."""
    labels = []
    for action in actions:
        if action["type"] == "skill":
            labels.append(action["command"])
        else:
            first = next((step["joints"].get(0) for step in action["steps"]
                          if step["joints"].get(0)), 0)
            labels.append("head_right" if first > 0 else "head_left" if first < 0 else "head_still")
    if labels != case["expected"]:
        return False
    return "first_seconds" not in case or actions[0].get("seconds") == case["first_seconds"]


def measure_plan(model, chain, text, language, valid_commands, case):
    from langchain_core.messages import HumanMessage

    counter = llm_utils.token_counter
    checkpoint = counter.snapshot()
    started = time.perf_counter()
    result = {"plan_status": "ok", "plan_correct": False, "failed_requests": 0}
    try:
        response = chain.invoke(
            [HumanMessage(content=f"[bahasa: {language}] {text}")],
            config={"configurable": {"session_id": "baseline"}},
        )
        counter.add(response, source="main", elapsed_s=time.perf_counter() - started)
        raw = response.content or response.additional_kwargs.get("reasoning_content", "")
        reply, actions = action_sequence.parse_reply(
            llm_utils.strip_thinking(raw), valid_commands, request_text=text,
        )
        prepared, warnings = action_sequence.prepare_actions(model, actions, valid_commands)
        result.update(reply=reply, actions=prepared, warnings=warnings,
                      plan_correct=bool(reply) and plan_matches(case, prepared))
    except action_sequence.ActionPlanError as error:
        result.update(plan_status="invalid", error_type=type(error).__name__)
    except Exception as error:
        result.update(plan_status="llm_error", error_type=type(error).__name__,
                      http_status=getattr(error, "status_code", None), failed_requests=1)
    result["llm_s"] = time.perf_counter() - started
    result["tokens"] = counter.delta(checkpoint)
    if result["failed_requests"]:
        # A failed request may have consumed tokens without returning usage.
        result["tokens"].update(usage_known=False, prompt_tokens=None,
                                completion_tokens=None, total_tokens=None, cached_tokens=None)
    result["token_calls"] = counter.records_since(checkpoint)
    result["tokens_last_60s_local"] = counter.rolling_summary()
    return result


def print_summary(summary, directory):
    accuracy = summary["accuracy"]["normalized"]
    timing = summary["timing"]["transcribe_s"]
    tokens = summary["tokens"]
    print(f"\n[Ringkasan] {summary['trials']} ucapan diuji")
    print(f"  Teks cocok: {accuracy['exact_count']}/{accuracy['scored_count']}")
    if accuracy["wer"] is not None:
        print(f"  Kesalahan kata (WER): {accuracy['wer']:.1%}; lebih rendah lebih baik")
    if timing["median"] is not None:
        print(f"  Transkripsi: median {timing['median']:.2f}s; p95 {timing['p95']:.2f}s")
    token_text = "tidak diketahui" if tokens["total_tokens"] is None else str(tokens["total_tokens"])
    print(f"  Token respons terukur: {token_text} ({tokens['calls']} respons)")
    if tokens["failed_requests"]:
        print(f"  Permintaan gagal tanpa hitungan lengkap: {tokens['failed_requests']}")
    if summary["plan_checked_count"]:
        print(f"  Rencana sesuai contoh: {summary['plan_correct_count']}/{summary['plan_checked_count']}")
    print(f"  Laporan: {directory / 'summary.json'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("tenang", "ramai"), required=True)
    parser.add_argument("--with-llm", action="store_true", help="ukur Groq juga; memakai kuota API")
    parser.add_argument("--repeat", type=int, default=1, choices=range(1, 6))
    parser.add_argument("--no-audio", action="store_true", help="simpan laporan tanpa berkas WAV")
    args = parser.parse_args(argv)

    stamp = datetime.now(timezone(timedelta(hours=7))).strftime("%Y%m%d-%H%M%S-%f")
    directory = PROJECT_ROOT / "logs" / "measurements" / f"{stamp}-{args.condition}"
    directory.mkdir(parents=True)
    print("[Uji] Robot tidak dihubungkan. Tidak ada gerakan atau pemutaran suara.")
    print("[Uji] Baca kalimat persis setelah 'Talk...'; jangan menyebut nomor contohnya.")
    print("[Uji] Pertahankan mikrofon, jarak, dan volume bicara pada kedua kondisi.")
    if not args.no_audio:
        print(f"[Uji] Rekaman WAV dan teks disimpan lokal di {directory}")
    if args.with_llm:
        print("[Uji] Groq akan dipanggil untuk transkripsi yang valid, termasuk perencana kepala.")

    records = []
    session = {"kind": "session", "condition": args.condition, "with_llm": args.with_llm,
               "record_audio": not args.no_audio, "cases": CASES,
               "supported_languages": list(speechtotext.SUPPORTED_LANGUAGES),
               "stt_language_mode": "auto_supported" if speechtotext.STT_LANGUAGE is None else speechtotext.STT_LANGUAGE,
               "python_version": sys.version.split()[0], "interrupted": False,
               "warmup_performed": False}
    model = chain = None
    try:
        speechtotext.initialize_transcriber()
        session["model_load_s"] = speechtotext.get_model_load_seconds()
        if session["model_load_s"] is not None:
            print(f"[Ukur suara] Pemuatan model: {session['model_load_s']:.2f}s")
        if args.with_llm:
            from dotenv import load_dotenv

            load_dotenv(PROJECT_ROOT / ".env")
            commands = json.loads((PROJECT_ROOT / "Commands.json").read_text(encoding="utf-8"))
            valid_commands = set(commands.values())
            model = llm_utils.get_model("groq")
            chain = build_chain(model, commands)

        write_json(directory / "session.json", session)
        for repetition in range(1, args.repeat + 1):
            for case in CASES:
                print(f"\n[Contoh {case['id']}/08; putaran {repetition}] Ucapkan: {case['text']}")
                if input("Enter untuk siap, atau q lalu Enter untuk selesai: ").strip().casefold() == "q":
                    session["interrupted"] = True
                    return
                wav_path = directory / f"{repetition}-{case['id']}.wav"
                capture_error = None
                try:
                    text, language = speechtotext.listen_and_transcribe(
                        audio_path=None if args.no_audio else wav_path,
                    )
                except Exception as error:
                    capture_error = error
                    text, language = None, None
                speech = speechtotext.get_last_measurement() or {}
                record = {"kind": "utterance", "case_id": case["id"], "repetition": repetition,
                          "condition": args.condition, "reference": case["text"], "speech": speech,
                          "first_trial_after_load": not records,
                          "audio_file": wav_path.name if wav_path.exists() else None,
                          "accuracy": {"raw": score_transcript(case["text"], speech.get("raw_text", "")),
                                       "normalized": score_transcript(case["text"], text or "")},
                          "tokens": llm_utils.TokenCounter().summary(), "llm_s": None,
                          "plan_status": "not_requested", "plan_correct": None}
                if args.with_llm and text:
                    record.update(measure_plan(model, chain, text, language, valid_commands, case))
                elif args.with_llm:
                    record.update(plan_status="no_speech", plan_correct=False)
                records.append(record)
                write_jsonl(directory / "utterances.jsonl", record)
                write_json(directory / "summary.json", summarize_records(records))
                print(f"[Hasil] {text or '(tidak ada teks)'}")
                if args.with_llm:
                    token_text = ("tidak diketahui" if record["tokens"]["total_tokens"] is None
                                  else str(record["tokens"]["total_tokens"]))
                    match = ("belum dinilai" if record["plan_correct"] is None
                             else "ya" if record["plan_correct"] else "tidak")
                    print(f"[Token ucapan] {token_text} | "
                          f"{record['tokens']['calls']} panggilan | status: {record['plan_status']} | "
                          f"rencana sesuai contoh: {match}")
                    if record.get("http_status") == 429:
                        print("[Uji] Groq sedang membatasi permintaan. Akhiri dengan q sebelum contoh berikutnya;")
                        print("      lanjutkan setelah waktu pemulihan akun, atau jalankan tanpa --with-llm.")
                if capture_error is not None:
                    raise capture_error
    except (KeyboardInterrupt, EOFError):
        session["interrupted"] = True
        print("\n[Uji] Pengukuran dihentikan; hasil yang selesai tetap disimpan.")
    except Exception as error:
        session["error_type"] = type(error).__name__
        print(f"[Uji] Pengukuran berhenti: {type(error).__name__}")
        raise
    finally:
        session["completed_trials"] = len(records)
        write_json(directory / "session.json", session)
        summary = summarize_records(records)
        write_json(directory / "summary.json", summary)
        print_summary(summary, directory)


if __name__ == "__main__":
    main()
