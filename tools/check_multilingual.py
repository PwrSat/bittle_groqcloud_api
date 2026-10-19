"""Uji pergantian bahasa otomatis tanpa robot, LLM, atau pemutaran suara."""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline_metrics import write_jsonl
import speechtotext


REFERENCE_CASES = (
    ("en", "Walk for three seconds, then sit down."),
    ("zh", "走三秒钟，然后坐下。"),
    ("hi", "तीन सेकंड चलो, फिर बैठ जाओ।"),
    ("es", "Camina durante tres segundos y luego siéntate."),
    ("ar", "امشِ لمدة ثلاث ثوانٍ، ثم اجلس."),
    ("fr", "Marche pendant trois secondes, puis assieds-toi."),
    ("pt", "Caminhe por três segundos e depois sente-se."),
    ("id", "Jalan selama tiga detik, lalu duduk."),
    ("ja", "三秒間歩いてから、座ってください。"),
    ("ko", "삼 초 동안 걷고 나서 앉아 주세요."),
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guided", action="store_true", help="tampilkan dan simpan 10 kalimat acuan")
    parser.add_argument("--languages", help="seleksi contoh --guided, kode dipisah koma; deteksi tetap otomatis")
    parser.add_argument("--gpu", action="store_true", help="gunakan Whisper medium tersimpan pada GPU CUDA")
    parser.add_argument("--samples", type=int, choices=range(1, 31),
                        help="jumlah ucapan; default jumlah contoh terpandu terpilih, selain itu 5")
    parser.add_argument("--no-audio", action="store_true", help="simpan metrik tanpa WAV")
    args = parser.parse_args(argv)
    references = REFERENCE_CASES
    if args.languages is not None:
        if not args.guided:
            parser.error("--languages hanya menyeleksi contoh dan memerlukan --guided")
        selected = {language.strip() for language in args.languages.split(",")}
        if not selected <= {language for language, _ in REFERENCE_CASES}:
            parser.error("--languages harus berisi kode: " + ",".join(language for language, _ in REFERENCE_CASES))
        references = tuple(case for case in REFERENCE_CASES if case[0] in selected)
    samples = args.samples if args.samples is not None else len(references) if args.guided else 5
    stamp = datetime.now(timezone(timedelta(hours=7))).strftime("%Y%m%d-%H%M%S-%f")
    directory = PROJECT_ROOT / "logs" / "measurements" / f"{stamp}-multilingual"
    directory.mkdir(parents=True)
    print("[Uji] Bahasa dikenali otomatis setiap ucapan. Coba berganti bahasa yang Anda kuasai.")
    print("[Uji] Tidak ada koneksi robot, panggilan LLM, atau pemutaran suara.")
    if args.guided:
        print("[Uji] Baca kalimat acuan persis setelah Talk...; bahasa tetap dideteksi otomatis.")
        if args.languages is not None:
            print("[Uji] Seleksi contoh: " + ", ".join(language for language, _ in references))
    print(f"[Uji] Metrik dan {'teks' if args.no_audio else 'rekaman WAV'} disimpan lokal di {directory}")
    if args.gpu:
        speechtotext.configure_transcriber(model="medium", device="cuda", compute_type="int8_float16")
    speechtotext.initialize_transcriber()
    try:
        for number in range(1, samples + 1):
            reference_language, reference = references[(number - 1) % len(references)]
            if args.guided:
                print(f"\n[Contoh {reference_language}] Ucapkan: {reference}")
            if input(f"\n[Ucapan {number}/{samples}] Enter untuk siap, q untuk selesai: ").strip().casefold() == "q":
                break
            text = language = None
            captured = False
            try:
                text, language = speechtotext.listen_and_transcribe(
                    audio_path=None if args.no_audio else directory / f"{number:02}.wav",
                )
                captured = True
            finally:
                measurement = speechtotext.get_last_measurement()
                if measurement is not None or args.guided:
                    row = {"sample": number, "speech": measurement}
                    if args.guided:
                        comparable = (captured and isinstance(measurement, dict)
                                      and measurement.get("status") == "ok"
                                      and isinstance(text, str) and bool(text.strip())
                                      and isinstance(language, str) and bool(language))
                        row.update(reference=reference, reference_language=reference_language,
                                   language_matches_reference=language == reference_language if comparable else None)
                    write_jsonl(directory / "utterances.jsonl", row)
            print(f"[Hasil] Bahasa: {language or 'belum dikenali'} | Teks: {text or '(kosong)'}")
    except (KeyboardInterrupt, EOFError):
        print("\n[Uji] Dihentikan.")
    finally:
        print(f"[Uji] Lokasi hasil: {directory}")


if __name__ == "__main__":
    main()
