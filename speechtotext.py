import io
import copy
import math
from pathlib import Path
import re
import time

from language_policy import SUPPORTED_LANGUAGES, select_language, reconcile_text_language

PROJECT_ROOT = Path(__file__).resolve().parent
# Aplikasi mendeteksi bahasa dalam SUPPORTED_LANGUAGES pada SETIAP ucapan.
# None tetap default; nilai eksplisit hanya untuk eksperimen/pengukuran lama.
STT_LANGUAGE = None
WHISPER_MODEL = "small"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
# Satu pencarian deterministik; fallback sampling berulang dapat melambat
# belasan detik tanpa memperbaiki ucapan yang sulit dikenali.
WHISPER_TEMPERATURE = 0.0
PAUSE_THRESHOLD_S = 1.3

# Petunjuk ejaan untuk decoder, bukan daftar jawaban yang wajib dihasilkan.
COMMAND_HINTS = (
    "Bittle, duduk, berdiri, jalan, maju, mundur, berhenti, diam, lompat, "
    "salam, tidur, istirahat, tengok kanan, tengok kiri, geleng kepala, "
    "udahan dulu, sit, stand, walk, stop, jump, hello"
)
# Petunjuk Latin di atas dapat terbaca sebagai romanisasi pada bahasa ini.
# Jangan menggantinya dengan daftar jawaban: decoder bisa menyalin daftar itu.
NON_LATIN_LANGUAGES = frozenset({"zh", "hi", "ar", "ja", "ko"})
LANGUAGE_CONFIDENCE_THRESHOLD = 0.6

_model = None
_recognizer = None
_microphone = None
_calibrated = False
_model_load_seconds = None
_last_measurement = None


def command_hints_for_language(language):
    """Pertahankan petunjuk lama hanya untuk kelompok aksara Latin."""
    return None if language in NON_LATIN_LANGUAGES else COMMAND_HINTS


def get_model_load_seconds():
    """Waktu pemuatan model terakhir; None jika belum pernah dimuat."""
    return _model_load_seconds


def get_last_measurement():
    """Salinan metrik rekaman terakhir tanpa menyimpan audio secara otomatis."""
    return copy.deepcopy(_last_measurement) if _last_measurement is not None else None


def _finite_number(value):
    """Ubah angka native/numpy menjadi nilai JSON, tanpa NaN atau infinity."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _audio_seconds(audio):
    frames = getattr(audio, "frame_data", None)
    sample_rate = _finite_number(getattr(audio, "sample_rate", None))
    sample_width = _finite_number(getattr(audio, "sample_width", None))
    if frames is None or not sample_rate or not sample_width:
        return None
    if sample_rate <= 0 or sample_width <= 0:
        return None
    return len(frames) / (sample_rate * sample_width)


def _log_measurement():
    def seconds(value):
        return f"{value:.2f}s" if value is not None else "n/a"

    print(
        "[Ukur suara] "
        f"tangkap={seconds(_last_measurement['capture_s'])}, "
        f"audio={seconds(_last_measurement['audio_s'])}, "
        f"transkripsi={seconds(_last_measurement['transcribe_s'])} "
        "(tangkap termasuk menunggu ucapan dan jeda akhir)"
    )


def _get_model():
    """Muat model sekali, hanya ketika pengenalan suara benar-benar dipakai."""
    global _model, _model_load_seconds
    if _model is None:
        started = time.perf_counter()
        try:
            if WHISPER_DEVICE == "cuda":
                from cuda_runtime import configure_cuda_libraries

                configure_cuda_libraries()
            from faster_whisper import WhisperModel

            print(f"[Sistem] Memuat Whisper {WHISPER_MODEL}...")
            print(f"[Sistem] Pemrosesan suara: {'GPU (CUDA)' if WHISPER_DEVICE == 'cuda' else 'CPU'}; {WHISPER_COMPUTE_TYPE}")
            options = {"device": WHISPER_DEVICE, "compute_type": WHISPER_COMPUTE_TYPE}
            if WHISPER_MODEL == "medium":
                options.update(download_root=str(PROJECT_ROOT / "models"), local_files_only=True)
            _model = WhisperModel(WHISPER_MODEL, **options)
        except Exception as error:
            if WHISPER_DEVICE == "cuda":
                raise RuntimeError("Whisper GPU belum dapat dimuat. Periksa dependensi CUDA di venv dan cache medium dalam folder models project.") from error
            raise
        finally:
            _model_load_seconds = time.perf_counter() - started
        print("[Sistem] Whisper siap.")
        if STT_LANGUAGE is None:
            print("[Sistem] Bahasa otomatis: " + ", ".join(SUPPORTED_LANGUAGES.values()))
    return _model


def configure_transcriber(*, model, device, compute_type=None):
    """Select an explicit startup profile before any model has been loaded."""
    global WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE
    if _model is not None:
        raise RuntimeError("Model suara sudah dimuat; jalankan proses baru untuk mengganti profil.")
    if not isinstance(model, str) or model not in {"small", "medium"}:
        raise ValueError("Model suara harus small atau medium.")
    allowed = {"cpu": {"int8", "int8_float32", "float32"},
               "cuda": {"int8_float16", "int8_float32", "int8", "float16", "float32"}}
    if not isinstance(device, str) or device not in allowed:
        raise ValueError("Perangkat suara harus cpu atau cuda.")
    if compute_type is None:
        compute_type = "int8" if device == "cpu" else "int8_float16"
    if not isinstance(compute_type, str) or compute_type not in allowed[device]:
        raise ValueError("Compute type tidak sesuai perangkat suara yang dipilih.")
    WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE = model, device, compute_type


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
        _recognizer.pause_threshold = PAUSE_THRESHOLD_S
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
        returned_language = "id"
        corrected = cleaned != "duduk" or language != returned_language
        return "duduk", returned_language, corrected

    return cleaned, language, False


def _decode_audio_for_detection(wav_data):
    from faster_whisper.audio import decode_audio

    return decode_audio(io.BytesIO(wav_data), sampling_rate=16000)


def transcribe_audio(wav_data, *, model=None):
    """Kenali satu WAV, dengan deteksi otomatis yang dibatasi bahasa dukungan.

    Deteksi dan transkripsi memakai model yang sama. Satu transkripsi penuh,
    tanpa menerjemahkan teks atau mencoba semua decoder bahasa bergantian.
    Probabilitas disimpan dari detektor asli, bukan angka 1 dari forced decode.
    """
    if model is None:
        model = _get_model()
    options = dict(
        temperature=WHISPER_TEMPERATURE, vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        condition_on_previous_text=False,
    )
    if STT_LANGUAGE is not None:
        segments, info = model.transcribe(
            io.BytesIO(wav_data), language=STT_LANGUAGE,
            hotwords=command_hints_for_language(STT_LANGUAGE), **options,
        )
        segments = list(segments)
        return {
            "segments": segments, "info": info,
            "text": " ".join(segment.text for segment in segments).strip(),
            "status": "ok", "language": info.language,
            "language_selected": info.language, "language_probability": None,
            "language_selected_probability": None,
            "language_detected": None, "language_detected_probability": None,
            "language_selection_reason": "explicit_experiment",
            "language_supported_mass": None, "language_candidates": [],
        }

    from faster_whisper.vad import VadOptions

    audio = _decode_audio_for_detection(wav_data)
    detected, probability, all_probs = model.detect_language(
        audio=audio, vad_filter=True, vad_parameters=VadOptions(min_silence_duration_ms=500),
    )
    choice = select_language(detected, probability, all_probs)
    selected = choice["language"]
    result = {
        "segments": [], "info": None, "text": "",
        "status": "unsupported_language" if selected is None else "ok",
        "language": selected, "language_selected": selected,
        "language_probability": choice["probability"],
        "language_selected_probability": choice["probability"],
        "language_detected": choice["detected_language"],
        "language_detected_probability": choice["detected_probability"],
        "language_selection_reason": choice["reason"],
        "language_supported_mass": choice["supported_mass"],
        "language_candidates": choice["alternatives"],
    }
    if selected is None:
        return result

    segments, info = model.transcribe(
        audio, language=selected, multilingual=False,
        hotwords=command_hints_for_language(selected), **options,
    )
    segments = list(segments)
    raw_text = " ".join(segment.text for segment in segments).strip()
    language, selected_probability, reason = reconcile_text_language(
        raw_text, selected, choice["probability"], all_probs,
        detected_probability=choice["detected_probability"],
    )
    result.update(segments=segments, info=info, text=raw_text, language=language,
                  language_probability=selected_probability,
                  language_selection_reason=choice["reason"] + ";" + reason)
    return result


def listen_and_transcribe(*, audio_path=None):
    """Rekam satu ucapan dan kembalikan (teks, bahasa), atau (None, None).

    Autodeteksi tetap menerima ucapan multibahasa. Probabilitas bahasa rendah
    bukan bukti bahwa isi teks salah, sehingga tidak sekadar mengganti tag
    bahasanya tanpa mengenali ulang audio.

    audio_path hanya dipakai untuk pengukuran yang secara eksplisit meminta
    salinan WAV asli. Pemakaian aplikasi biasa tidak menyimpan rekaman.
    """
    global _calibrated, _last_measurement

    _last_measurement = {
        "model": WHISPER_MODEL,
        "device": WHISPER_DEVICE,
        "compute_type": WHISPER_COMPUTE_TYPE,
        "status": "capture_error",
        "load_model_s": get_model_load_seconds(),
        "calibration_s": 0.0,
        "capture_s": None,
        "audio_s": None,
        "transcribe_s": None,
        "raw_text": None,
        "normalized_text": None,
        "language_detected": None,
        "language_returned": None,
        "language_probability": None,
        "language_probability_scope": "language_only",
        "language_selected": None,
        "language_selected_probability": None,
        "language_returned_probability": None,
        "language_supported_mass": None,
        "language_selection_reason": None,
        "language_candidates": [],
        "supported_languages": list(SUPPORTED_LANGUAGES),
        "mean_avg_logprob": None,
        "mean_no_speech_prob": None,
        "mean_compression_ratio": None,
        "decoder_temperatures": [],
        "decoder_temperature_setting": WHISPER_TEMPERATURE,
        "pause_threshold_s": PAUSE_THRESHOLD_S,
        "segment_count": 0,
        "energy_threshold": None,
        "error_type": None,
        "error_stage": None,
    }

    # Gagal memuat model/perangkat harus terlihat oleh pemanggil, bukan menjadi
    # pengulangan tanpa henti karena loop utama mengabaikan hasil kosong.
    try:
        model = _get_model()
    except BaseException as error:
        _last_measurement.update(
            load_model_s=get_model_load_seconds(),
            error_type=type(error).__name__, error_stage="model_load",
        )
        raise
    _last_measurement["load_model_s"] = get_model_load_seconds()

    recognizer = None
    try:
        recognizer, microphone = _get_audio_devices()
        with microphone as source:
            if not _calibrated:
                print("[Sistem] Kalibrasi mikrofon: diam sebentar...")
                started = time.perf_counter()
                try:
                    recognizer.adjust_for_ambient_noise(source, duration=1)
                finally:
                    _last_measurement["calibration_s"] = time.perf_counter() - started
                _calibrated = True

            # Tampilkan tanda siap SETELAH kalibrasi, agar awal kata tidak
            # terpakai sebagai sampel suara latar dan hilang dari rekaman.
            print("Talk...")
            started = time.perf_counter()
            try:
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=10)
            finally:
                _last_measurement["capture_s"] = time.perf_counter() - started
        _last_measurement["audio_s"] = _audio_seconds(audio)
    except BaseException as error:
        _last_measurement.update(
            error_type=type(error).__name__, error_stage="capture",
        )
        raise
    finally:
        if recognizer is not None:
            _last_measurement["energy_threshold"] = _finite_number(
                getattr(recognizer, "energy_threshold", None)
            )

    started = None
    try:
        wav_data = audio.get_wav_data()
        if audio_path is not None:
            Path(audio_path).write_bytes(wav_data)
        started = time.perf_counter()
        decoded = transcribe_audio(wav_data, model=model)
        segments = decoded["segments"]
        elapsed = time.perf_counter() - started
        _last_measurement["transcribe_s"] = elapsed
        _last_measurement.update(
            language_detected=decoded["language_detected"],
            language_probability=decoded["language_detected_probability"],
            language_selected=decoded["language_selected"],
            language_selected_probability=decoded["language_selected_probability"],
            language_returned_probability=decoded["language_probability"],
            language_supported_mass=decoded["language_supported_mass"],
            language_selection_reason=decoded["language_selection_reason"],
            language_candidates=decoded["language_candidates"],
        )
        if decoded["status"] == "unsupported_language":
            _last_measurement["status"] = "unsupported_language"
            print("[Sistem] Bahasa belum cukup sesuai daftar dukungan. Ucapkan ulang dengan lebih jelas.")
            _log_measurement()
            return None, None
        _last_measurement["segment_count"] = len(segments)
        for attribute, key in (
            ("avg_logprob", "mean_avg_logprob"),
            ("no_speech_prob", "mean_no_speech_prob"),
            ("compression_ratio", "mean_compression_ratio"),
        ):
            values = [
                value for segment in segments
                if (value := _finite_number(getattr(segment, attribute, None)))
                is not None
            ]
            _last_measurement[key] = sum(values) / len(values) if values else None
        _last_measurement["decoder_temperatures"] = [
            value for segment in segments
            if (value := _finite_number(getattr(segment, "temperature", None)))
            is not None
        ]
        raw_text = decoded["text"]
        selected_probability = decoded["language_probability"]
        text, detected_lang, corrected = normalize_transcript(
            raw_text, decoded["language"], selected_probability if selected_probability is not None else 1.0
        )
        if detected_lang != decoded["language"]:
            _last_measurement["language_returned_probability"] = None
        _last_measurement.update(
            status="ok" if text else "empty",
            raw_text=raw_text, normalized_text=text,
            language_returned=detected_lang if text else None,
        )

        if not text:
            print(f"[Sistem] Whisper: tidak ada teks ucapan valid [{elapsed:.1f}s]")
            _log_measurement()
            return None, None

        if corrected:
            print(f"[Sistem] Koreksi perintah: {raw_text!r} -> {text!r}")

        if STT_LANGUAGE is not None:
            language_note = "bahasa ditetapkan, tanpa deteksi otomatis"
        elif detected_lang != decoded["language_detected"]:
            original_probability = decoded["language_detected_probability"]
            probability_note = (f" {original_probability:.0%}" if original_probability is not None else "")
            language_note = f"deteksi awal: {decoded['language_detected']}{probability_note}; diperiksa otomatis"
        else:
            language_note = (f"keyakinan bahasa: {selected_probability:.0%}"
                             if selected_probability is not None else "keyakinan bahasa tidak tersedia")
        print(
            f"[Sistem] Transkripsi selesai [{elapsed:.1f}s]. "
            f"Bahasa: {detected_lang} ({language_note})"
        )
        _log_measurement()
        return text, detected_lang

    except Exception as e:
        if started is not None and _last_measurement["transcribe_s"] is None:
            _last_measurement["transcribe_s"] = time.perf_counter() - started
        _last_measurement.update(
            status="decode_error", error_type=type(e).__name__,
            error_stage="decode" if started is not None else "audio_encoding",
        )
        print(f"Error saat pengenalan suara: {e}")
        _log_measurement()
        return None, None


if __name__ == "__main__":
    hasil, lang = listen_and_transcribe()
    print(f"Hasil transkripsi: {hasil} (bahasa: {lang})")
