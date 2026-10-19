"""Bandingkan Whisper small/medium pada WAV tersimpan, tanpa mikrofon atau robot.

Default --dry hanya memeriksa rekaman. --run memakai cache lokal;
tambahkan --download untuk mengizinkan unduhan ke folder models project.
"""

import argparse
from datetime import datetime, timedelta, timezone
import gc
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import wave

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _project_path(value):
    try:
        path = Path(value).resolve()
        path.relative_to(PROJECT_ROOT.resolve())
        return path
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError("Berkas rekaman dan cache harus berada di dalam project.") from error


def prepare_audio_cases(session, languages=None):
    from tools.check_language_understanding import load_session_cases, resolve_session

    source = _project_path(resolve_session(session))
    cases = load_session_cases(source)
    samples = {
        f"recorded-{row['sample']:02d}-{row['reference_language']}": row["sample"]
        for row in (json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip())
    }
    selected = []
    for case in cases:
        if languages is not None and case["reference_language"] not in languages:
            continue
        sample = samples[case["id"]]
        path = _project_path(source.parent / f"{sample:02}.wav")
        try:
            audio = path.read_bytes()
            with wave.open(io.BytesIO(audio), "rb") as wav:
                frames, rate = wav.getnframes(), wav.getframerate()
                width, channels = wav.getsampwidth(), wav.getnchannels()
                if not frames or not rate or wav.getcomptype() != "NONE":
                    raise ValueError("WAV kosong/tidak berupa PCM")
                if len(wav.readframes(frames)) != frames * width * channels:
                    raise ValueError("WAV terpotong")
        except (OSError, EOFError, wave.Error, ValueError) as error:
            raise ValueError(f"Rekaman {sample:02}.wav hilang/rusak. Pilih sesi yang menyimpan audio WAV.") from error
        selected.append({**case, "sample": sample, "wav_path": path,
                         "audio_bytes": audio, "audio_s": frames / rate})
    if not selected:
        raise ValueError("Tidak ada rekaman untuk seleksi --languages ini.")
    return source, selected


def _whisper_model_class():
    from faster_whisper import WhisperModel

    return WhisperModel


def load_model(name, *, device, compute_type, allow_download):
    cache = _project_path(PROJECT_ROOT / "models")
    # Set before importing the HF backend: downloads and Xet data stay in project.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_XET_CACHE"] = str(cache / ".xet")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if device == "cuda":
        from cuda_runtime import configure_cuda_libraries

        configure_cuda_libraries()
    factory = _whisper_model_class()
    options = {"device": device, "compute_type": compute_type}
    if name == "small":
        try:
            return factory("small", local_files_only=True, **options)
        except Exception:
            pass  # An absent global cache may be supplied by the project cache.
    if allow_download:
        cache.mkdir(parents=True, exist_ok=True)
    return factory(name, download_root=str(cache), local_files_only=not allow_download, **options)


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def transcribe_case(wav_data, model):
    import speechtotext as stt

    if stt.STT_LANGUAGE is not None:
        raise ValueError("Perbandingan memerlukan deteksi bahasa otomatis (STT_LANGUAGE=None).")
    started = time.perf_counter()
    decoded = stt.transcribe_audio(wav_data, model=model)
    elapsed = time.perf_counter() - started
    raw = decoded["text"]
    probability = decoded["language_probability"]
    text, language, _ = stt.normalize_transcript(
        raw, decoded["language"], probability if probability is not None else 1.0,
    )
    result = {
        "status": decoded["status"] if decoded["status"] != "ok" else "ok" if text else "empty",
        "raw_text": raw, "normalized_text": text, "transcribe_s": elapsed,
        "language_detected": decoded["language_detected"],
        "language_probability": decoded["language_detected_probability"],
        "language_selected": decoded["language_selected"],
        "language_selected_probability": decoded["language_selected_probability"],
        "language_returned": language if text else None,
        "language_returned_probability": probability if language == decoded["language"] else None,
        "language_supported_mass": decoded["language_supported_mass"],
        "language_selection_reason": decoded["language_selection_reason"],
        "language_candidates": decoded["language_candidates"],
        "decoder_temperatures": [_number(segment.temperature) for segment in decoded["segments"]],
        "segment_count": len(decoded["segments"]),
    }
    for attribute in ("avg_logprob", "no_speech_prob", "compression_ratio"):
        values = [_number(getattr(segment, attribute, None)) for segment in decoded["segments"]]
        values = [value for value in values if value is not None]
        result["mean_" + attribute] = sum(values) / len(values) if values else None
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry", action="store_true", help="periksa rencana tanpa memuat model (default)")
    mode.add_argument("--run", action="store_true", help="jalankan transkripsi ulang pada WAV yang sama")
    parser.add_argument("--session", default="latest", help="folder/utterances.jsonl/latest")
    parser.add_argument("--languages", help="seleksi contoh acuan; deteksi bahasa tetap otomatis")
    parser.add_argument("--models", nargs="+", choices=("small", "medium"), default=["small", "medium"])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--compute-type", help="default int8 pada CPU, int8_float16 pada CUDA")
    parser.add_argument("--download", action="store_true", help="izinkan unduhan hanya ke models project")
    args = parser.parse_args(argv)
    if args.download and not args.run:
        parser.error("--download memerlukan --run")
    from tools.check_language_understanding import LANGUAGES

    languages = None if args.languages is None else {item.strip() for item in args.languages.split(",")}
    if languages is not None and not languages <= set(LANGUAGES):
        parser.error("--languages harus berisi kode: " + ",".join(LANGUAGES))
    try:
        source, cases = prepare_audio_cases(args.session, languages)
        _project_path(PROJECT_ROOT / "models")
    except ValueError as error:
        parser.error(str(error))
    models = list(dict.fromkeys(args.models))
    compute = args.compute_type or ("int8" if args.device == "cpu" else "int8_float16")
    print("[Uji] WAV tersimpan yang sama; bahasa tetap otomatis, tanpa mikrofon/robot/API Groq.")
    print(f"[Rencana] Model: {', '.join(models)}; perangkat: {args.device}; compute: {compute}")
    print(f"[Sumber] {source}")
    for case in cases:
        print(f"[Audio] {case['sample']:02}.wav; acuan {case['reference_language']}; {case['audio_s']:.2f}s")
    if not args.run:
        print("[Dry] Tidak ada model yang dimuat atau diunduh. Gunakan --run untuk memulai.")
        return 0

    stamp = datetime.now(timezone(timedelta(hours=7))).strftime("%Y%m%d-%H%M%S-%f")
    directory = _project_path(PROJECT_ROOT / "logs" / "measurements" / f"{stamp}-whisper-comparison")
    directory.mkdir(parents=True)
    report = {"kind": "whisper_model_comparison", "source_session": str(source),
              "device": args.device, "compute_type": compute, "download_allowed": args.download,
              "scoring": "manual review of native text; no automatic semantic/WER score", "models": []}
    failed = False
    try:
        for name in models:
            model = None
            result = {"model": name, "status": "ok", "model_load_s": None, "records": []}
            report["models"].append(result)
            try:
                started = time.perf_counter()
                model = load_model(name, device=args.device, compute_type=compute, allow_download=args.download)
                result["model_load_s"] = time.perf_counter() - started
                print(f"[Model] {name} siap dalam {result['model_load_s']:.2f}s")
                for case in cases:
                    record = {key: str(value) if isinstance(value, Path) else value
                              for key, value in case.items()
                              if key not in {"audio_bytes", "text", "language"}}
                    record.update(previous_asr_text=case["text"],
                                  previous_asr_language=case["language"])
                    try:
                        record.update(transcribe_case(case["audio_bytes"], model))
                    except Exception as error:
                        record.update(status="error", error_type=type(error).__name__, transcribe_s=None)
                        result["status"], failed = "error", True
                    result["records"].append(record)
                    print("[Hasil] " + json.dumps({"model": name, "sample": case["sample"],
                          **{key: record.get(key) for key in ("status", "normalized_text", "language_returned", "transcribe_s")}}, ensure_ascii=False))
            except Exception as error:
                result.update(status="error", error_type=type(error).__name__)
                failed = True
                print(f"[Gagal] {name}: cache/model/perangkat belum tersedia. Gunakan --run --download untuk cache project; periksa CUDA jika memilih GPU.", file=sys.stderr)
            finally:
                del model
                gc.collect()
                (directory / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except KeyboardInterrupt:
        report["interrupted"] = True
        failed = True
    finally:
        output = directory / "summary.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[Laporan] {output}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
