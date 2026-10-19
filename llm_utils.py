import os
import re
import math
import time
from collections.abc import Mapping

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
        return ChatGroq(model="openai/gpt-oss-120b", temperature=0)

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
    """Catat metadata token dan waktu panggilan, tanpa menyimpan isi percakapan.

    ``snapshot()`` mengembalikan indeks untuk ``delta(snapshot)`` dan
    ``records_since(snapshot)``. Nilai token yang tidak dilaporkan provider
    ditulis ``None`` pada catatan/ringkasan; atribut lama prompt/completion
    tetap menjumlahkan angka yang diketahui demi kompatibilitas.

    ``rolling_summary()`` hanya mengamati panggilan dalam proses ini, bukan
    sisa kuota provider. Jendela menggunakan waktu monoton saat respons dicatat,
    dengan batas bawah terbuka: (sekarang - window_s, sekarang].
    """

    def __init__(self, clock=None):
        self.prompt = 0
        self.completion = 0
        self.calls = 0
        self._clock = clock or time.monotonic
        self._records = []

    @staticmethod
    def _mapping(value):
        return value if isinstance(value, Mapping) else {}

    @staticmethod
    def _count(*values):
        """Ambil hitungan pertama yang valid; nol berbeda dari tidak diketahui."""
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
                continue
            if value >= 0:
                return int(value)
        return None

    def add(self, response, *, source="llm", elapsed_s=None):
        """Tambahkan respons; tetap return (prompt, completion) dengan fallback 0.

        ``elapsed_s`` adalah waktu invoke yang diukur pemanggil. Tidak ada
        estimasi token dari teks apabila metadata provider tidak tersedia.
        """
        meta = self._mapping(getattr(response, "response_metadata", None))
        usage = self._mapping(meta.get("token_usage"))
        modern = self._mapping(getattr(response, "usage_metadata", None))
        p = self._count(usage.get("prompt_tokens"), modern.get("input_tokens"),
                        meta.get("prompt_eval_count"))
        c = self._count(usage.get("completion_tokens"), modern.get("output_tokens"),
                        meta.get("eval_count"))
        total = p + c if p is not None and c is not None else self._count(
            usage.get("total_tokens"), modern.get("total_tokens")
        )
        cached = self._count(
            self._mapping(usage.get("prompt_tokens_details")).get("cached_tokens"),
            self._mapping(modern.get("input_token_details")).get("cache_read"),
        )
        if (isinstance(elapsed_s, bool) or not isinstance(elapsed_s, (int, float))
                or not math.isfinite(elapsed_s) or elapsed_s < 0):
            elapsed_s = None
        else:
            elapsed_s = float(elapsed_s)
        self._records.append({
            "source": source if isinstance(source, str) and source else "llm",
            "elapsed_s": elapsed_s,
            "completed_at_monotonic_s": self._clock(),
            "prompt_tokens": p,
            "completion_tokens": c,
            "total_tokens": total,
            "cached_tokens": cached,
            "usage_known": p is not None and c is not None,
        })
        self.prompt += p if p is not None else 0
        self.completion += c if c is not None else 0
        self.calls += 1
        return (p if p is not None else 0), (c if c is not None else 0)

    def snapshot(self):
        """Checkpoint integer sebelum sebuah giliran atau kelompok panggilan."""
        return len(self._records)

    def records_since(self, snapshot=0):
        """Salinan catatan JSON-serializable sejak checkpoint, tanpa teks/secrets."""
        if (isinstance(snapshot, bool) or not isinstance(snapshot, int)
                or not 0 <= snapshot <= len(self._records)):
            raise ValueError("snapshot harus berasal dari counter ini")
        return [dict(record) for record in self._records[snapshot:]]

    @staticmethod
    def _summarize(records):
        def total_if_known(field):
            values = [record[field] for record in records]
            return sum(values) if all(value is not None for value in values) else None

        result = {"calls": len(records)}
        for field in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens"):
            result[field] = total_if_known(field)
        result["usage_known"] = all(record["usage_known"] for record in records)
        result["missing_usage_calls"] = sum(not record["usage_known"] for record in records)
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            result["known_" + field] = sum(record[field] or 0 for record in records)
        return result

    def delta(self, snapshot):
        """Ringkasan semua panggilan sejak checkpoint, termasuk planner kepala."""
        return self._summarize(self.records_since(snapshot))

    def summary(self):
        """Ringkasan sesi; angka parsial juga tersedia sebagai known_*_tokens."""
        return self._summarize(self._records)

    def rolling_summary(self, window_s=60.0):
        """Pengamatan jendela waktu lokal; bukan perhitungan sisa kuota server."""
        if (isinstance(window_s, bool) or not isinstance(window_s, (int, float))
                or not math.isfinite(window_s) or window_s <= 0):
            raise ValueError("window_s harus angka positif yang finite")
        now = self._clock()
        return self._summarize([
            record for record in self._records
            if now - window_s < record["completed_at_monotonic_s"] <= now
        ])

    @property
    def total(self):
        return self.prompt + self.completion


token_counter = TokenCounter()
