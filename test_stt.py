import speech_recognition as sr

r = sr.Recognizer()
mic = sr.Microphone()

with mic as source:
    print("Ngomong sekarang...")
    r.adjust_for_ambient_noise(source, duration=1)
    audio = r.listen(source, phrase_time_limit=5)

try:
    text = r.recognize_google(audio, language="id-ID")
    print("Hasil:", text)
except Exception as e:
    print("Error detail:", type(e).__name__, e)