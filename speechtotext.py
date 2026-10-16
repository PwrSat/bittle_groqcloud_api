import io
import re
import time

# None = deteksi otomatis multibahasa. Pilih "id" jika seluruh sesi berbahasa
# Indonesia; ini menghindari salah tebak bahasa pada perintah satu kata.
STT_LANGUAGE = None
WHISPER_MODEL = "small"

# Petunjuk ejaan untuk decoder, bukan daftar jawaban yang wajib dihasilkan.
COMMAND_HINTS = (
    "Bittle, duduk, berdiri, jalan, maju, mundur, berhenti, diam, lompat, "
    "salam, tidur, istirahat, tengok kanan, tengok kiri, geleng kepala, "
    "udahan dulu, sit, stand, walk, stop, jump, hello"
)
LANGUAGE_CONFIDENCE_THRESHOLD = 0.6

_model = None
_recognizer = None
_microphone = None
_calibrated = False


def _get_model():
    """Muat model sekali, hanya ketika pengenalan suara benar-benar dipakai."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        print(f"[Sistem] Memuat Whisper {WHISPER_MODEL}...")
        _model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        print("[Sistem] Whisper siap.")
    return _model


def initialize_transcriber():
    """Siapkan model sebelum koneksi BLE, tanpa membuka mikrofon.

    Pada lingkungan Windows proyek ini, memuat model native CTranslate2
    setelah backend BLE/WinRT diimpor dapat menyebabkan access violation.
    Impor modul ini tetap ringan; aplikasi memanggil initializer secara eksplisit.
    """
    return _get_model()


def _get_audio_devices():
    """Pertahankan ambang suara antar giliran, termasuk adaptasi suara latar."""
    global _recognizer, _microphone
    import speech_recognition as sr

    if _recognizer is None:
        _recognizer = sr.Recognizer()
        _recognizer.pause_threshold = 1.0
    if _microphone is None:
        _microphone = sr.Microphone()
    return _recognizer, _microphone


def normalize_transcript(text, language, confidence):
    """Saring hasil kosong dan koreksi hanya salah dengar satu perintah utuh.

    Tidak memakai fuzzy matching: kalimat seperti 'jangan duduk' atau cerita
    yang mengandung 'duh duk' harus tetap utuh. Bahasa lain yang terdeteksi
    dengan yakin juga tidak diubah menjadi perintah Indonesia.
    """
    cleaned = " ".join(text.split())
    if not any(char.isalnum() for char in cleaned):
        return "", language, False

    words = " ".join(re.findall(r"[^\W\d_]+", cleaned.casefold()))
    # Alias ini berasal dari salah transkripsi yang terlihat pada log pengguna.
    # Pencocokan dibatasi ke seluruh ujaran, termasuk pengulangan perintah.
    sit_alias = r"(?:duduk|duh duk|doudouk)"
    if (
        (language == "id" or confidence < LANGUAGE_CONFIDENCE_THRESHOLD)
        and not any(char.isdigit() for char in cleaned)
        and re.fullmatch(rf"{sit_alias}(?: {sit_alias})*", words)
    ):
        corrected = cleaned != "duduk" or language != "id"
        return "duduk", "id", corrected

    return cleaned, language, False


def listen_and_transcribe():
    """Rekam satu ucapan dan kembalikan (teks, bahasa), atau (None, None).

    Autodeteksi tetap menerima ucapan multibahasa. Probabilitas bahasa rendah
    bukan bukti bahwa isi teks salah, sehingga tidak sekadar mengganti tag
    bahasanya tanpa mengenali ulang audio.
    """
    global _calibrated

    # Gagal memuat model/perangkat harus terlihat oleh pemanggil, bukan menjadi
    # pengulangan tanpa henti karena loop utama mengabaikan hasil kosong.
    model = _get_model()
    recognizer, microphone = _get_audio_devices()

    with microphone as source:
        if not _calibrated:
            print("[Sistem] Kalibrasi mikrofon: diam sebentar...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            _calibrated = True

        # Tampilkan tanda siap SETELAH kalibrasi, agar awal kata tidak
        # terpakai sebagai sampel suara latar dan hilang dari rekaman.
        print("Talk...")
        audio = recognizer.listen(source, timeout=None, phrase_time_limit=10)

    try:
        wav_stream = io.BytesIO(audio.get_wav_data())
        started = time.perf_counter()
        segments, info = model.transcribe(
            wav_stream,
            language=STT_LANGUAGE,
            hotwords=COMMAND_HINTS,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=False,
        )
        raw_text = " ".join(segment.text for segment in segments).strip()
        text, detected_lang, corrected = normalize_transcript(
            raw_text, info.language, info.language_probability
        )
        elapsed = time.perf_counter() - started

        if not text:
            print(f"[Sistem] Whisper: tidak ada teks ucapan valid [{elapsed:.1f}s]")
            return None, None

        if corrected:
            print(f"[Sistem] Koreksi perintah: {raw_text!r} -> {text!r}")

        if STT_LANGUAGE is not None:
            language_note = "bahasa ditetapkan, tanpa deteksi otomatis"
        elif detected_lang != info.language:
            language_note = (
                f"deteksi awal: {info.language}, "
                f"keyakinan bahasa: {info.language_probability:.0%}"
            )
        else:
            language_note = f"keyakinan bahasa: {info.language_probability:.0%}"
        print(
            f"[Sistem] Transkripsi selesai [{elapsed:.1f}s]. "
            f"Bahasa: {detected_lang} ({language_note})"
        )
        return text, detected_lang

    except Exception as e:
        print(f"Error saat pengenalan suara: {e}")
        return None, None


if __name__ == "__main__":
    hasil, lang = listen_and_transcribe()
    print(f"Hasil transkripsi: {hasil} (bahasa: {lang})")
