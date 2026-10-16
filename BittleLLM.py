import os
import json
import time
from pathlib import Path

# Matikan tracing LangSmith (tidak dipakai, cuma bikin error 403 & delay)
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from speechtotext import initialize_transcriber, listen_and_transcribe
from Text2Speech import text_to_speech_stream
from robot_link import create_robot
from llm_utils import get_model, strip_thinking, is_exit_command, token_counter
import action_sequence

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# ================================================================
# PENGATURAN UTAMA
# ================================================================
LLM_PROVIDER = "groq"        # "groq" (cloud) | "glm" | "kimi" (lokal, butuh Ollama + GPU kuat)
CONNECTION_MODE = "ble"      # "ble" (Bluetooth) | "serial" (kabel USB)
ROBOT_BLE_ADDRESS = None     # isi MAC address robot (misal "AA:BB:CC:DD:EE:FF") biar
                             # tidak perlu scan 8 detik tiap start; None = scan otomatis
MAX_HISTORY_MESSAGES = 20    # batasi riwayat chat (~10 giliran) biar hemat token
COMMANDS_FILE = PROJECT_ROOT / "Commands.json"
ACTION_DRY_RUN = False       # True = semua aksi hanya DITAMPILKAN, tanpa dikirim ke robot.

# ================================================================
# SETUP LLM
# ================================================================
# Muat Whisper sebelum LLM dan backend BLE/WinRT agar inisialisasi native
# tidak mengalami access violation pada Windows. Mikrofon belum dibuka.
initialize_transcriber()
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

SYSTEM_PROMPT = action_sequence.build_system_prompt(json.loads(commands_json_text))
# Semua JSON contoh harus di-escape agar tidak menjadi placeholder LangChain.
escaped_system_prompt = SYSTEM_PROMPT.replace("{", "{{").replace("}", "}}")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", escaped_system_prompt),
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
    print("Bittle terhubung. Tunggu 'Talk...' sebelum mulai berbicara.")

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
            p_tok, c_tok = token_counter.add(response)
            print(f"[Token] masuk={p_tok} keluar={c_tok} | total sesi={token_counter.total} "
                  f"({token_counter.calls} panggilan)")

            # --- Ambil teks jawaban (fallback ke field cadangan kalau content kosong) ---
            raw_text = response.content
            if not raw_text or not raw_text.strip():
                raw_text = (response.additional_kwargs.get("reasoning_content", "")
                            or response.additional_kwargs.get("thinking", ""))

            cleaned = strip_thinking(raw_text)
            try:
                description, actions = action_sequence.parse_reply(
                    cleaned or "", VALID_COMMANDS, request_text=user_input
                )
            except action_sequence.ActionPlanError as e:
                print(f"[Rencana ditolak] {e}")
                text_to_speech_stream("Maaf, rangkaian gerakannya belum bisa kubaca. Coba lagi ya.", lang="id")
                continue

            # --- Eksekusi ---
            if not description:
                print(f"[Peringatan] AI tidak menghasilkan jawaban final [{_elapsed:.1f}s].")
                text_to_speech_stream("Maaf, aku belum sempat mikir jawabannya. Coba lagi ya.", lang="id")

            elif actions:
                try:
                    # Rencanakan setiap aksi kepala secara terpisah dan periksa
                    # SELURUH rangkaian sebelum robot melakukan gerakan pertama.
                    prepared, warnings = action_sequence.prepare_actions(model, actions, VALID_COMMANDS)
                    for w in warnings:
                        print(f"[Validasi gerakan] {w}")
                    print(f"[Bittle]: {description}  ({len(prepared)} aksi)  [AI {_elapsed:.1f}s]")
                    text_to_speech_stream(description, lang=detected_lang)
                    action_sequence.execute_actions(robot, prepared, VALID_COMMANDS, dry_run=ACTION_DRY_RUN)
                except Exception as e:
                    print(f"[Error rangkaian] {e}")
                    text_to_speech_stream("Maaf, rangkaian gerakannya dihentikan. Coba lagi ya.", lang="id")

            else:
                print(f"[Bittle]: {description}  (tidak ada command)  [{_elapsed:.1f}s]")
                text_to_speech_stream(description, lang=detected_lang)

    except KeyboardInterrupt:
        print("\nMenghentikan Bittle... mengembalikan ke posisi idle.")
        try:
            robot.send("kup")
        except Exception as e:
            print(f"[Error stop] {e}")
    finally:
        robot.close()
        print(f"[Token] Ringkasan sesi: masuk={token_counter.prompt} keluar={token_counter.completion} "
              f"total={token_counter.total} dalam {token_counter.calls} panggilan")
        print("Program berhenti dengan bersih.")
