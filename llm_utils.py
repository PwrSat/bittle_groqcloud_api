import os
import re

# Matikan tracing LangSmith (tidak dipakai; kalau .env masih berisi
# LANGCHAIN_TRACING_V2=true dari repo lama, ini bikin error 403 + delay).
# Diset paling awal & dipaksa, karena load_dotenv() tidak menimpa env yang sudah ada.
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"


def strip_thinking(text: str) -> str:
    """
    Hilangkan bagian "thinking" / chain-of-thought dari respons model
    reasoning (GLM-4.7-Flash, Kimi-VL, dll) sebelum diproses lebih lanjut
    (ekstraksi command & text-to-speech). Model non-reasoning seperti
    Groq/Llama tidak menghasilkan format ini, jadi fungsi ini aman
    dipanggil untuk semua provider tanpa efek samping.
    """
    if not text:
        return text

    # Format GLM: "Thinking...\n...\n...done thinking."
    text = re.sub(r"Thinking\.\.\..*?\.\.\.done thinking\.", "", text, flags=re.DOTALL)

    # Format Kimi-VL: "◁think▷...◁/think▷"
    text = re.sub(r"◁think▷.*?◁/think▷", "", text, flags=re.DOTALL)

    # Format umum reasoning model lain (fallback): "<think>...</think>"
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # PENTING: Ollama untuk model reasoning (GLM dkk) kadang cuma kirim
    # tag PENUTUP tanpa tag pembuka (karena field "thinking" digabung di
    # awal content oleh langchain-ollama) -> ambil teks SETELAH tag
    # penutup terakhir, buang semua "proses berpikir" di depannya
    if "</think>" in text.lower():
        idx = text.lower().rfind("</think>")
        text = text[idx + len("</think>"):]

    # Terakhir: bersihin token tag mentah apapun yang masih nyempil
    # (misal tag PEMBUKA tanpa penutup sama sekali, kayak Kimi-VL yang
    # kadang nulis "◁think▷" di depan tapi nggak pernah nutup)
    text = text.replace("◁think▷", "").replace("◁/think▷", "")
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)

    return text.strip()


def extract_final_reply(text: str):
    """
    Ekstrak kalimat balasan final + command dari respons LLM.

    PENTING: pakai match ##...## PALING AKHIR (bukan yang pertama
    ditemukan), karena reasoning model (GLM/Kimi) kadang mengutip ulang
    instruksi system prompt (yang berisi contoh literal "##command##")
    selama proses "mikir" -- kalau ambil match pertama, yang kepilih
    malah placeholder dari kutipan instruksi, bukan command asli yang
    ditulis di baris paling akhir.

    Juga robust terhadap reasoning yang TIDAK dibungkus tag apapun
    (tidak ke-cover oleh strip_thinking()) dengan cara cuma ambil baris
    TERAKHIR sebagai kalimat balasan, bukan seluruh teks mentah.

    Return: (description, dogcommand) -- dogcommand bisa None kalau
    tidak ada command terdeteksi sama sekali.
    """
    if not text:
        return "", None

    matches = list(re.finditer(r"##(.*?)##", text))

    if not matches:
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        description = lines[-1] if lines else text.strip()
        description = _clean_markdown_artifacts(description)
        return description, None

    last_match = matches[-1]
    dogcommand = last_match.group(1).strip().rstrip(".")

    text_before = text[:last_match.start()]
    lines_before = [l.strip() for l in text_before.strip().splitlines() if l.strip()]
    description = lines_before[-1] if lines_before else ""
    description = _clean_markdown_artifacts(description)

    return description, dogcommand


def _clean_markdown_artifacts(line: str) -> str:
    """Bersihin sisa format markdown yang kadang nyempil di baris terakhir
    reasoning model, misal '6. **Output Generation:** teks asli' -> 'teks asli'."""
    line = re.sub(r"^[\d]+[\.\)]\s*", "", line)
    line = re.sub(r"\*\*[^*]*\*\*:?\s*", "", line)
    line = re.sub(r"^[\*\-]\s*", "", line)
    return line.strip().strip('"')


def get_model(provider: str):
    """
    Factory untuk memilih LLM provider.

    provider:
        "groq" -> GPT-OSS 120B via GroqCloud (API, cloud, cepat)
        "glm"  -> GLM-4.7-Flash via Ollama (lokal, butuh GPU)
        "kimi" -> Kimi-VL-A3B via Ollama (lokal, butuh GPU)
    """
    provider = provider.lower().strip()

    if provider == "groq":
        from langchain_groq import ChatGroq
        # Catatan: "llama-3.3-70b-versatile" sudah pindah ke tier
        # Enterprise di Groq (butuh "Contact Sales"), diganti ke
        # openai/gpt-oss-120b yang masih di tier Production/free.
        return ChatGroq(model="openai/gpt-oss-120b")

    elif provider == "glm":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model="glm-4.7-flash",
            num_predict=2048,
            num_ctx=8192,
            reasoning=False,
        )

    elif provider == "kimi":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model="richardyoung/kimi-vl-a3b-thinking",
            num_predict=2048,
            num_ctx=8192,
            reasoning=False,
        )

    else:
        raise ValueError(
            f"Provider tidak dikenal: '{provider}'. Pilih 'groq', 'glm', atau 'kimi'."
        )


_EXIT_KEYWORDS = [
    "matikan", "mati kan", "mati aja", "mati dulu",
    "udahan", "sudahan", "sudah dulu", "cukup dulu",
    "selesai main", "berhenti main", "stop main",
    "keluar program", "tutup program", "stop program", "matikan program",
    "off dulu", "off aja",
    "segitu aja dulu", "segini aja dulu", "itu aja dulu", "itu saja dulu",
    "mainnya udahan", "main nya udahan", "main nya sudah",
    "bye bittle", "dadah bittle",
]


def is_exit_command(text: str) -> bool:
    """Cek apakah kalimat user menandakan keinginan mengakhiri sesi
    (misal 'oke udahan dulu', 'matikan', 'sudah dulu main nya')."""
    if not text:
        return False
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in _EXIT_KEYWORDS)


class TokenCounter:
    """Catat pemakaian token selama sesi (dibaca dari metadata respons LLM)."""

    def __init__(self):
        self.prompt = 0
        self.completion = 0
        self.calls = 0

    def add(self, response):
        meta = getattr(response, "response_metadata", None) or {}
        usage = meta.get("token_usage") or {}
        # Groq: token_usage{prompt_tokens, completion_tokens}; Ollama: prompt_eval_count/eval_count
        p = usage.get("prompt_tokens") or meta.get("prompt_eval_count") or 0
        c = usage.get("completion_tokens") or meta.get("eval_count") or 0
        self.prompt += p
        self.completion += c
        self.calls += 1
        return p, c

    @property
    def total(self):
        return self.prompt + self.completion


token_counter = TokenCounter()