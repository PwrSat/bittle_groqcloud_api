import os
import re
import json
import time

# Matikan tracing LangSmith (tidak dipakai, cuma bikin error 403 & delay)
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from speechtotext import listen_and_transcribe
from Text2Speech import text_to_speech_stream
from robot_link import create_robot
from llm_utils import get_model, strip_thinking, extract_final_reply, is_exit_command

load_dotenv()

# ================================================================
# PENGATURAN UTAMA
# ================================================================
LLM_PROVIDER = "groq"        # "groq" (cloud) | "glm" | "kimi" (lokal, butuh Ollama + GPU kuat)
CONNECTION_MODE = "ble"      # "ble" (Bluetooth) | "serial" (kabel USB)
ROBOT_BLE_ADDRESS = None     # isi MAC address robot (misal "AA:BB:CC:DD:EE:FF") biar
                             # tidak perlu scan 8 detik tiap start; None = scan otomatis
MAX_HISTORY_MESSAGES = 10    # batasi riwayat chat (~10 giliran) biar hemat token
COMMANDS_FILE = "Commands.json"

# ================================================================
# SETUP LLM
# ================================================================
model = get_model(LLM_PROVIDER)
print(f"[Sistem] Provider LLM: {LLM_PROVIDER}")

# ================================================================
# SETUP ROBOT (Bluetooth / Serial)
# ================================================================
print(f"[Sistem] Menghubungkan ke robot via {CONNECTION_MODE}...")
try:
    if CONNECTION_MODE == "ble":
        robot = create_robot("ble", address=ROBOT_BLE_ADDRESS)
    else:
        robot = create_robot("serial")
except Exception as e:
    print(f"[Error] Gagal terhubung ke robot: {e}")
    raise SystemExit(1)

# ================================================================
# RIWAYAT CHAT (dibatasi supaya tidak membengkak)
# ================================================================
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    history = store[session_id]
    if len(history.messages) > MAX_HISTORY_MESSAGES:
        history.messages = history.messages[-MAX_HISTORY_MESSAGES:]
    return history


# ================================================================
# SYSTEM PROMPT (Commands.json di-embed langsung)
# ================================================================
with open(COMMANDS_FILE, "r", encoding="utf-8") as f:
    commands_json_text = f.read()

# Kode command yang VALID -> dipakai buat validasi sebelum kirim ke robot
VALID_COMMANDS = set(json.loads(commands_json_text).values())

# Escape { } supaya JSON tidak dianggap placeholder template oleh LangChain
escaped_commands_json = commands_json_text.replace("{", "{{").replace("}", "}}")

SYSTEM_PROMPT = (
    "Kamu adalah Bittle, robot anjing peliharaan yang ramah dan antusias. "
    "Pengguna berbicara padamu lewat suara dan kamu menjawab sebagai robot anjing, "
    "sekaligus menerjemahkan maksud kalimatnya menjadi command gerakan. "
    "ATURAN BAHASA: jawab SELALU dalam bahasa yang SAMA dengan bahasa pengguna "
    "(Indonesia dijawab Indonesia, Inggris dijawab Inggris, dst). Di awal tiap pesan "
    "ada tag seperti '[bahasa: en]' yang menunjukkan bahasa terdeteksi; gunakan itu "
    "sebagai acuan dan JANGAN tulis tag itu dalam jawabanmu. "
    "ATURAN PANJANG: jawaban HARUS singkat, maksimal 1 kalimat pendek (5-12 kata), "
    "natural seperti anjing peliharaan. Beri jawaban lebih panjang HANYA jika pengguna "
    "secara eksplisit meminta penjelasan atau cerita. "
    "ATURAN STOP: jika pengguna minta berhenti/diam/stop (bahasa apapun), SELALU pakai "
    "command ##kup##. JANGAN pernah pakai command gerakan seperti mundur untuk itu. "
    "ATURAN COMMAND: HANYA gunakan command dari daftar JSON berikut (kode command "
    "tidak pernah diterjemahkan): "
    + escaped_commands_json +
    " Contoh: pengguna bilang 'ayo lompat' -> 'Ayo lompat! ##kjmp##'; "
    "pengguna bilang \"let's jump\" -> \"Let's jump! ##kjmp##\". "
    "Selalu taruh command di bagian PALING AKHIR jawaban dengan format ##kode##, "
    "tanpa teks apapun setelahnya. Jika kalimat pengguna bukan permintaan gerakan "
    "(misalnya cuma ngobrol), tetap jawab singkat dan pilih satu command yang cocok "
    "dengan suasananya, atau ##khi## untuk sapaan."
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

# ================================================================
# LOOP UTAMA
# ================================================================
if __name__ == "__main__":
    print("Bittle siap! Silakan mulai berbicara.")

    try:
        while True:
            user_input, detected_lang = listen_and_transcribe()

            if not user_input:
                # Tidak ada suara valid (diam/noise) -> lewati, jangan kirim ke AI
                continue

            print(f"[Kamu bilang]: {user_input}  (bahasa: {detected_lang})")

            # --- Perintah keluar ("udahan", "matikan", dst) ---
            if is_exit_command(user_input):
                print("[Sistem] Perintah keluar terdeteksi. Menghentikan Bittle...")
                farewell = "Baik bosku, sampai jumpa lagi!"
                print(f"[Bittle]: {farewell}")
                text_to_speech_stream(farewell, lang="id")
                robot.send("kup")
                break

            # --- Kirim ke LLM (dengan tag bahasa terdeteksi) ---
            tagged_input = f"[bahasa: {detected_lang}] {user_input}"

            _start = time.time()
            try:
                response = with_message_history.invoke(
                    [HumanMessage(content=tagged_input)],
                    config=config,
                )
            except Exception as e:
                # Misal kena rate limit Groq (429) atau internet putus
                print(f"[Error LLM] {e}")
                text_to_speech_stream("Maaf, aku lagi error. Coba lagi ya.", lang="id")
                continue
            _elapsed = time.time() - _start

            #cek penggunaan token
            usage = response.response_metadata.get("token_usage", {})
            print(f"[Token] masuk={usage.get('prompt_tokens')} keluar={usage.get('completion_tokens')}")

            # --- Ambil teks jawaban (fallback ke field cadangan kalau content kosong) ---
            raw_text = response.content
            if not raw_text or not raw_text.strip():
                raw_text = (response.additional_kwargs.get("reasoning_content", "")
                            or response.additional_kwargs.get("thinking", ""))

            cleaned = strip_thinking(raw_text)
            description, dogcommand = extract_final_reply(cleaned) if cleaned else ("", None)

            # --- Eksekusi ---
            if not description:
                print(f"[Peringatan] AI tidak menghasilkan jawaban final [{_elapsed:.1f}s].")
                text_to_speech_stream("Maaf, aku belum sempat mikir jawabannya. Coba lagi ya.", lang="id")

            elif dogcommand and dogcommand in VALID_COMMANDS:
                print(f"[Bittle]: {description}  (command: {dogcommand})  [{_elapsed:.1f}s]")
                robot.send(dogcommand)                       # gerak dulu...
                text_to_speech_stream(description, lang=detected_lang)  # ...sambil/lalu ngomong

            elif dogcommand:
                # Ada ##...## tapi bukan command valid -> jangan kirim ke robot
                print(f"[Bittle]: {description}  (command TIDAK VALID: {dogcommand!r}, diabaikan)  [{_elapsed:.1f}s]")
                text_to_speech_stream(description, lang=detected_lang)

            else:
                print(f"[Bittle]: {description}  (tidak ada command)  [{_elapsed:.1f}s]")
                text_to_speech_stream(description, lang=detected_lang)

    except KeyboardInterrupt:
        print("\nMenghentikan Bittle... mengembalikan ke posisi idle.")
        robot.send("kup")
    finally:
        robot.close()
        print("Program berhenti dengan bersih.")