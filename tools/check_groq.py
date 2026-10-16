"""Cek API Groq secara manual memakai GROQ_API_KEY dari .env proyek."""

from pathlib import Path


def main():
    from dotenv import load_dotenv
    from groq import Groq

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    client = Groq()
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": "Halo, kamu siapa?"}],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
