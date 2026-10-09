from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import io

# Mapping kode bahasa hasil deteksi Whisper -> kode bahasa yang didukung
# gTTS. Kebanyakan sama persis (ISO 639-1), tapi beberapa perlu
# disesuaikan (misal Mandarin "zh" -> "zh-CN" di gTTS).
WHISPER_TO_GTTS_LANG = {
    "zh": "zh-CN",
    "jw": "jv",  # javanese edge case kalau kedetect
}

# Fallback kalau kode bahasa yang terdeteksi ternyata tidak didukung gTTS
DEFAULT_LANG = "id"


def _resolve_gtts_lang(lang_code):
    if not lang_code:
        return DEFAULT_LANG
    return WHISPER_TO_GTTS_LANG.get(lang_code, lang_code)


def text_to_speech_stream(text, lang=DEFAULT_LANG):
    """
    Ubah teks jadi suara pakai gTTS (gratis, tanpa API key).

    lang: kode bahasa ISO 639-1 (default "id"). Kalau kode dari Whisper
    tidak didukung gTTS, otomatis fallback ke DEFAULT_LANG.
    """
    if not text or not text.strip():
        print("Error: teks kosong, tidak ada yang diucapkan.")
        return

    gtts_lang = _resolve_gtts_lang(lang)

    try:
        tts = gTTS(text=text, lang=gtts_lang)

        audio_stream = io.BytesIO()
        tts.write_to_fp(audio_stream)
        audio_stream.seek(0)

        sound = AudioSegment.from_file(audio_stream, format="mp3")
        play(sound)
    except Exception as e:
        # Kalau bahasa yang diminta ternyata tidak didukung gTTS sama
        # sekali, coba fallback ke default sebelum benar-benar menyerah
        if gtts_lang != DEFAULT_LANG:
            print(f"Error TTS untuk bahasa '{gtts_lang}' ({e}), coba fallback ke '{DEFAULT_LANG}'...")
            try:
                tts = gTTS(text=text, lang=DEFAULT_LANG)
                audio_stream = io.BytesIO()
                tts.write_to_fp(audio_stream)
                audio_stream.seek(0)
                sound = AudioSegment.from_file(audio_stream, format="mp3")
                play(sound)
            except Exception as e2:
                print(f"Error saat text-to-speech (fallback juga gagal): {e2}")
        else:
            print(f"Error saat text-to-speech: {e}")