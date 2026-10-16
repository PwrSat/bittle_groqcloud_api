"""Cek Google Speech Recognition; aplikasi utama memakai Whisper."""


def main():
    import speech_recognition as sr

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    with microphone as source:
        print("Kalibrasi mikrofon: diam sebentar...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Ngomong sekarang...")
        audio = recognizer.listen(source, phrase_time_limit=5)
    try:
        text = recognizer.recognize_google(audio, language="id-ID")
        print("Hasil:", text)
    except Exception as error:
        print("Error detail:", type(error).__name__, error)


if __name__ == "__main__":
    main()
