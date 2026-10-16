"""Daftar mikrofon; pemeriksaan perangkat hanya dilakukan lewat main()."""


def main():
    import speech_recognition as sr

    print("Daftar microphone yang terdeteksi di sistem ini:\n")
    microphones = sr.Microphone.list_microphone_names()
    for index, name in enumerate(microphones):
        print(f"[{index}] {name}")
    print(f"\nTotal: {len(microphones)} device ditemukan")
    print("\nAplikasi memakai mikrofon default sistem.")


if __name__ == "__main__":
    main()
