import sys
import time

from speechtotext import listen_and_transcribe
from Text2Speech import text_to_speech_stream
from ardSerial import *
import speech_recognition as sr
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from llm_utils import get_model, strip_thinking, extract_final_reply, is_exit_command
import re

load_dotenv()

# ================================================================
# PILIH PROVIDER LLM DI SINI: "groq", "glm", atau "kimi"
# - "groq" -> cloud, cepat, butuh internet + GROQ_API_KEY
# - "glm"  -> lokal via Ollama, butuh GPU (RTX 5080), tanpa API key
# - "kimi" -> lokal via Ollama, butuh GPU (RTX 5080), tanpa API key
# ================================================================
LLM_PROVIDER = "groq"

model = get_model(LLM_PROVIDER)
print(f"[Sistem] Menggunakan provider LLM: {LLM_PROVIDER}")

store = {}
goodPorts = {}
connectPort(goodPorts)

MAX_HISTORY_MESSAGES = 20  # ~10 giliran percakapan (user+AI), batasi biar
                            # tidak numpuk tanpa batas dan bikin context
                            # window kepenuhan di sesi yang panjang


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    history = store[session_id]
    # Trim riwayat lama kalau udah kepanjangan
    if len(history.messages) > MAX_HISTORY_MESSAGES:
        history.messages = history.messages[-MAX_HISTORY_MESSAGES:]
    return history


# Baca Commands.json DULUAN (sebelum bikin system prompt), supaya bisa
# di-embed langsung ke instruksi -> tidak perlu lagi priming call
# terpisah yang bikin startup lambat & numpuk riwayat chat dari awal.
file_path = 'Commands.json'
with open(file_path, 'r', encoding='utf-8') as file:
    file_content = file.read()

import json
VALID_COMMANDS = set(json.loads(file_content).values())

# PENTING: escape { dan } di isi JSON dulu, karena ChatPromptTemplate
# menganggap { } sebagai placeholder template (kayak {nama_variable}).
# Tanpa di-escape, JSON kita (penuh kurung kurawal) bikin LangChain
# salah kira itu variable yang perlu diisi -> error "Expected mapping
# type... Received list".
escaped_commands_json = file_content.replace("{", "{{").replace("}", "}}")

SYSTEM_PROMPT = (
    "Hi, saya sedang mengembangkan robot anjing yang bisa diprogram. Saya membangun software untuk mengontrol robot ini dari jarak jauh, dan saya juga ingin ngobrol dengan robot ini. Saya akan memberikan beberapa kalimat ke robot ini dan kamu akan menjawab sebagai robot anjing. Namamu adalah Bittle. PENTING SOAL BAHASA: kamu HARUS selalu merespons dalam BAHASA YANG SAMA dengan bahasa yang saya pakai bertanya (kalau saya bicara Bahasa Inggris, jawab Bahasa Inggris; kalau Bahasa Jepang, jawab Bahasa Jepang; dst) -- JANGAN pernah terjemahkan ke bahasa lain, ikuti persis bahasa input saya. Di awal tiap pesan saya, akan ada tag seperti '[bahasa: en]' yang menandakan bahasa yang terdeteksi dari suara saya -- gunakan ini sebagai acuan bahasa jawabanmu, dan JANGAN sertakan tag ini dalam responsmu. ATURAN PANJANG JAWABAN: jawabanmu HARUS singkat, maksimal 1 kalimat pendek (sekitar 5-12 kata) sebelum command. Jangan bertele-tele, jangan jelaskan alasan atau perasaan panjang lebar. Cukup respons singkat dan natural seperti anjing peliharaan yang antusias. Hanya berikan jawaban lebih panjang KALAU saya secara eksplisit minta kamu menjelaskan sesuatu atau bercerita. ATURAN COMMAND 'STOP': kalau saya bilang 'stop', 'berhenti', 'diam', atau minta kamu berhenti bergerak (dalam bahasa apapun), SELALU gunakan command ##kup## (posisi berdiri/idle). JANGAN PERNAH gunakan command gerakan seperti retreat/mundur untuk permintaan berhenti. Kamu akan menerjemahkan MAKSUD kalimat saya menjadi command yang sesuai berdasarkan command set yang kita punya (nama command selalu dalam format aslinya, TIDAK diterjemahkan), dan memberikan string command tersebut ke saya. Berikut daftar command yang valid dalam format json -- HANYA gunakan command dari daftar ini: "
    + escaped_commands_json +
    " Contoh: kalau saya bilang 'ayo lompat' atau 'let's jump', kamu akan merespons dengan bahasa yang sama seperti 'Ayo lompat! ##kjmp##' atau 'Let's jump! ##kjmp##'. Selalu letakkan command di bagian PALING AKHIR responsmu, ditulis sebagai ##command##, tanpa ada teks setelahnya. Contoh: ##ksit##"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
chain = prompt | model
config = {"configurable": {"session_id": "firstChat"}}
with_message_history = RunnableWithMessageHistory(chain, get_session_history)

print("Bittle siap! Silakan mulai berbicara.")

if __name__ == "__main__":

    try:
        while True:
            user_input, detected_lang = listen_and_transcribe()

            if not user_input:
                # Tidak ada suara valid yang tertangkap (diam/noise) -> skip,
                # jangan dikirim ke AI
                continue

            print(f"[Kamu bilang]: {user_input}  (bahasa: {detected_lang})")

            if is_exit_command(user_input):
                print("[Sistem] Perintah keluar terdeteksi. Menghentikan Bittle...")
                farewell = "Baik bosku, sampai jumpa lagi!"
                print(f"[Bittle]: {farewell}")
                text_to_speech_stream(farewell, lang="id")
                try:
                    send(goodPorts, ["kup", 1])
                except Exception:
                    pass
                break

            # Sisipkan tag bahasa terdeteksi di awal pesan, supaya LLM tahu
            # harus jawab dalam bahasa apa (lihat instruksi di system prompt)
            tagged_input = f"[bahasa: {detected_lang}] {user_input}"

            _start = time.time()
            response = with_message_history.invoke(
                [HumanMessage(content=tagged_input)],
                config=config,
            )
            _elapsed = time.time() - _start

            raw_text = response.content
            if not raw_text or not raw_text.strip():
                # Kadang reasoning=False tidak 100% dihormati model -> teks
                # "mikir" nyasar ke additional_kwargs, bukan ke content.
                # Coba selamatkan dari sana sebagai fallback.
                fallback = response.additional_kwargs.get("reasoning_content", "") \
                    or response.additional_kwargs.get("thinking", "")
                raw_text = fallback

            command = strip_thinking(raw_text)
            description, dogcommand = extract_final_reply(command) if command else ("", None)

            if not description:
                # Kejadian ini kalau model kehabisan token budget di
                # tengah "mikir" sebelum sempat nulis jawaban final,
                # atau context window kepenuhan
                print(f"[Peringatan] AI tidak menghasilkan jawaban final [{_elapsed:.1f}s].")
                print(f"  content: {response.content[:300]!r}")
                print(f"  additional_kwargs: {str(response.additional_kwargs)[:300]!r}")
                description = "Maaf, aku belum sempat mikir jawabannya. Coba lagi ya."
                text_to_speech_stream(description, lang="id")
            elif dogcommand and dogcommand in VALID_COMMANDS:
                print(f"[Bittle]: {description}  (command: {dogcommand})  [{_elapsed:.1f}s]")
                text_to_speech_stream(description, lang=detected_lang)

                task = [dogcommand, 1]
                send(goodPorts, task)

            elif dogcommand:
                # AI nulis sesuatu di antara ##...## tapi bukan kode command
                # yang valid (misal literal "command", typo, dsb) -> jangan
                # kirim ke robot, cukup tetap ngomong aja
                print(f"[Bittle]: {description}  (command TIDAK VALID: "
                      f"{dogcommand!r}, diabaikan)  [{_elapsed:.1f}s]")
                text_to_speech_stream(description, lang=detected_lang)

            else:
                print(f"[Bittle]: {description}  (tidak ada command terdeteksi)  [{_elapsed:.1f}s]")
                text_to_speech_stream(description, lang=detected_lang)

    except KeyboardInterrupt:
        print("\nMenghentikan Bittle... mengembalikan ke posisi idle.")
        try:
            send(goodPorts, ["kup", 1])
        except Exception:
            pass
    finally:
        for port_obj in list(goodPorts.keys()):
            try:
                port_obj.Close_Engine()
            except Exception:
                pass
        print("Program berhenti dengan bersih.")