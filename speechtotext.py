import io
import time
import speech_recognition as sr
from faster_whisper import WhisperModel
from Text2Speech import text_to_speech_stream

# ================================================================
# Load model Whisper SEKALI SAJA saat modul di-import (bukan tiap kali
# transcribe), supaya tidak reload model setiap giliran bicara.
# ================================================================
_model = WhisperModel("medium", device="cpu", compute_type="int8")

# Bahasa yang "diharapkan" dipakai user -> kalau Whisper nebak salah
# satu dari ini, PERCAYA walau confidence-nya rendah. Kalimat pendek
# (misal "STOP!", "Can you walk?") itu SECARA ALAMI confidence-nya
# rendah di Whisper walau tebakannya tetap akurat -- jadi nolak
# berdasarkan angka confidence doang malah bikin salah.
EXPECTED_LANGUAGES = {"id", "en", "ja", "zh", "ko", "ar", "fr", "de", "es"}

# Threshold ini CUMA dipakai kalau bahasa yang terdeteksi di LUAR
# daftar expected (kemungkinan besar salah tebak total, kayak kasus
# "Konichiwa" -> Swahili) -- butuh confidence tinggi baru dipercaya
UNEXPECTED_LANG_THRESHOLD = 0.6

# Bahasa terakhir yang terdeteksi dengan YAKIN, dipakai sebagai fallback
_last_confident_lang = "id"


def listen_and_transcribe():
    """
    Rekam suara dari mic, transkripsi pakai Whisper dengan AUTO-DETECT
    bahasa (mendukung banyak bahasa, tidak dikunci ke Bahasa Indonesia
    saja) supaya robot bisa diajak bicara multibahasa.

    Return: (text, lang_code) atau (None, None) kalau tidak ada suara
    valid yang tertangkap. lang_code contoh: "id", "en", "ja", "zh", dst
    (kode ISO 639-1 yang dideteksi Whisper).
    """
    global _last_confident_lang

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    recognizer.pause_threshold = 1.0

    with microphone as source:
        print("Talk...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source, timeout=None, phrase_time_limit=10)

    try:
        wav_bytes = audio.get_wav_data()
        wav_stream = io.BytesIO(wav_bytes)

        _t0 = time.time()
        segments, info = _model.transcribe(
            wav_stream,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text for segment in segments).strip()
        _elapsed = time.time() - _t0

        if not text:
            print(f"[Sistem] Whisper: tidak ada suara valid [{_elapsed:.1f}s]")
            return None, None

        detected_lang = info.language
        confidence = info.language_probability

        # Logic baru: percaya tebakan Whisper kalau bahasanya masuk akal
        # (ada di EXPECTED_LANGUAGES), WALAU confidence-nya rendah --
        # karena kalimat pendek emang secara alami confidence-nya rendah
        # tapi tebakannya biasanya tetap benar.
        if detected_lang in EXPECTED_LANGUAGES:
            _last_confident_lang = detected_lang
        elif confidence >= UNEXPECTED_LANG_THRESHOLD:
            # Bahasa "aneh"/di luar expected, tapi confidence-nya tinggi
            # -> tetap percaya, mungkin beneran bahasa itu
            _last_confident_lang = detected_lang
        else:
            # Bahasa aneh DAN confidence rendah -> kemungkinan besar
            # salah tebak total (kayak kasus Konichiwa -> Swahili)
            print(f"[Sistem] Deteksi bahasa '{detected_lang}' di luar ekspektasi, "
                  f"confidence rendah ({confidence:.0%}) -> pakai bahasa terakhir "
                  f"yang yakin: '{_last_confident_lang}'")
            detected_lang = _last_confident_lang

        print(f"[Sistem] Transkripsi selesai [{_elapsed:.1f}s]. "
              f"Bahasa: {detected_lang} (confidence: {confidence:.0%})")
        return text, detected_lang

    except Exception as e:
        print(f"Error saat transkripsi Whisper: {e}")
        return None, None


if __name__ == "__main__":
    hasil, lang = listen_and_transcribe()
    print(f"Hasil transkripsi: {hasil} (bahasa: {lang})")