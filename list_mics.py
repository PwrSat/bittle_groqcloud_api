import speech_recognition as sr

print("Daftar microphone yang terdeteksi di sistem ini:\n")
mics = sr.Microphone.list_microphone_names()

for i, name in enumerate(mics):
    print(f"[{i}] {name}")

print(f"\nTotal: {len(mics)} device ditemukan")
print("\nMic default yang dipakai sr.Microphone() tanpa parameter biasanya index 0,")
print("tapi ini bisa SALAH kalau Windows default device-nya bukan mic yang kamu mau.")