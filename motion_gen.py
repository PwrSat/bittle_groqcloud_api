"""
motion_gen.py -- LLM membantu menyusun gerakan motor tertentu (micro-movement)

Alur:
  kalimat user -> LLM menyusun RENCANA GERAKAN (JSON) -> VALIDASI KETAT
  (clamp sudut, batasi motor/jumlah langkah/durasi) -> kirim ke robot secara HALUS
  (interpolasi bertahap) -> kembali netral

PRINSIP KEAMANAN: LLM tidak pernah mengirim apa pun langsung ke robot. Semua
keluaran LLM harus lolos validate_plan() dulu: motor yang tidak ada di
JOINT_LIMITS dibuang, sudut dipotong ke batas aman, langkah & durasi dibatasi.
"""
import json
import math
import re
import time

# ================================================================
# KONFIGURASI KEAMANAN -- ubah di sini, bukan di prompt
# ================================================================
# {index_motor: (sudut_min, sudut_maks)}
# Bittle: 0 = kepala (menoleh kiri/kanan), 8-11 = bahu, 12-15 = lutut.
# AWAL: hanya kepala yang diizinkan, karena paling aman (tidak bikin robot jatuh).
# Motor kaki memakai sudut ABSOLUT terhadap posisi kalibrasi, bukan relatif ke pose
# berdiri -- jangan diaktifkan sebelum kamu paham pose robot saat itu.
JOINT_LIMITS = {
    0: (-60, 60),
}

JOINT_DESC = {
    0: "kepala: menoleh kiri-kanan saja. KANAN = sudut POSITIF, KIRI = sudut NEGATIF "
       "(jangan terbalik). Kepala TIDAK bisa mengangguk/menunduk.",
}

# Arah fisik robot kamu terbalik terhadap konvensi di atas, jadi dibalik di sini
JOINT_SIGN = {0: -1}

MAX_STEPS = 8
HOLD_RANGE = (0.2, 2.0)       # detik tahan per langkah
MAX_TOTAL_SECONDS = 12.0      # total durasi TAHAN satu gerakan
NEUTRAL_ANGLE = 0             # posisi netral kepala

# --- Kehalusan gerak ---
SMOOTH = True                 # False = lompat langsung ke sudut tujuan (kaku)
SPEED_DEG_PER_S = 120         # kecepatan menoleh; turunkan (misal 80) kalau mau lebih pelan
UPDATE_HZ = 20                # berapa kali perintah sudut dikirim per detik
MIN_MOVE_S = 0.15             # durasi minimum satu perpindahan

# Kirim perintah 'i' kosong di akhir untuk melepas kontrol manual kepala.
# Kalau kepala terasa kaku/aneh SETELAH gerakan, coba ubah ke False.
RELEASE_HEAD_AFTER = True
HEAD_NEUTRAL_SETTLE_S = 0.2
HEAD_RELEASE_SETTLE_S = 0.3  # waktu firmware memproses 'i' sebelum perintah berikutnya

RIGHT_WORDS = ("kanan", "right", "migi", "右")
LEFT_WORDS = ("kiri", "left", "hidari", "左")


# ================================================================
# 1. PROMPT KE LLM
# ================================================================
def build_motion_prompt():
    joint_lines = "\n".join(
        f"- motor {idx}: {JOINT_DESC.get(idx, 'motor')} | sudut {lo} sampai {hi}"
        for idx, (lo, hi) in JOINT_LIMITS.items()
    )
    return (
        "Kamu adalah perancang gerakan untuk robot anjing Bittle. Ubah permintaan "
        "pengguna menjadi urutan langkah gerakan motor.\n\n"
        "Motor yang boleh digerakkan:\n" + joint_lines + "\n\n"
        "Balas HANYA dengan JSON valid, tanpa penjelasan, tanpa markdown, format:\n"
        '{"steps": [{"joints": {"0": 40}, "hold": 0.5}, ...]}\n'
        "- 'joints': kunci = nomor motor (string), nilai = sudut (angka bulat)\n"
        "- 'hold': lama menahan posisi itu dalam detik\n"
        f"- maksimal {MAX_STEPS} langkah; hold antara {HOLD_RANGE[0]} dan {HOLD_RANGE[1]}\n"
        "- jangan pakai motor di luar daftar. Gerakan akan otomatis kembali netral di akhir.\n\n"
        "Contoh 'gelengkan kepala':\n"
        '{"steps": [{"joints": {"0": -40}, "hold": 0.4}, {"joints": {"0": 40}, "hold": 0.4}, '
        '{"joints": {"0": -40}, "hold": 0.4}, {"joints": {"0": 40}, "hold": 0.4}]}\n'
        "Contoh 'tengok ke kanan pelan-pelan' (kanan = POSITIF):\n"
        '{"steps": [{"joints": {"0": 20}, "hold": 0.5}, {"joints": {"0": 45}, "hold": 1.5}]}\n'
        "Contoh 'tengok ke kiri' (kiri = NEGATIF):\n"
        '{"steps": [{"joints": {"0": -45}, "hold": 1.0}]}'
    )


# ================================================================
# 2. PARSE + VALIDASI (fungsi murni, bisa dites tanpa robot/LLM)
# ================================================================
def parse_plan(text):
    """Ambil objek JSON pertama dari teks LLM (toleran terhadap ```json ... ```)."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def validate_plan(plan):
    """
    Return (steps_bersih, peringatan).
    steps_bersih = [{"joints": {int: int}, "hold": float}, ...]
    """
    warnings = []
    if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list):
        return [], ["rencana tidak punya daftar 'steps'"]

    clean = []
    total = 0.0
    for i, step in enumerate(plan["steps"][:MAX_STEPS]):
        if not isinstance(step, dict) or not isinstance(step.get("joints"), dict):
            warnings.append(f"langkah {i + 1} dibuang (format salah)")
            continue

        joints = {}
        for key, val in step["joints"].items():
            try:
                idx, ang = int(key), float(val)
            except (TypeError, ValueError):
                warnings.append(f"langkah {i + 1}: motor/sudut '{key}':{val} tidak valid")
                continue
            if idx not in JOINT_LIMITS:
                warnings.append(f"langkah {i + 1}: motor {idx} tidak diizinkan, dibuang")
                continue
            lo, hi = JOINT_LIMITS[idx]
            clamped = max(lo, min(hi, ang))
            if clamped != ang:
                warnings.append(f"langkah {i + 1}: motor {idx} sudut {ang:g} dipotong ke {clamped:g}")
            joints[idx] = int(round(clamped))

        if not joints:
            continue

        try:
            hold = float(step.get("hold", 0.5))
        except (TypeError, ValueError):
            hold = 0.5
        hold = max(HOLD_RANGE[0], min(HOLD_RANGE[1], hold))

        if total + hold > MAX_TOTAL_SECONDS:
            warnings.append(f"langkah {i + 1} dan sesudahnya dibuang (melebihi {MAX_TOTAL_SECONDS:g} detik)")
            break
        total += hold
        clean.append({"joints": joints, "hold": hold})

    if len(plan["steps"]) > MAX_STEPS:
        warnings.append(f"langkah dipotong ke {MAX_STEPS}")
    return clean, warnings


def enforce_direction(steps, request_text):
    """
    LLM kadang terbalik soal kiri/kanan. Kalau permintaan menyebut SATU arah saja
    (kanan ATAU kiri), pastikan gerakan kepala pertama searah dengan itu; kalau tidak,
    balik tanda semua sudut kepala. (Permintaan yang menyebut keduanya dibiarkan.)
    """
    text = (request_text or "").lower()
    has_r = any(w in text for w in RIGHT_WORDS)
    has_l = any(w in text for w in LEFT_WORDS)
    if has_r == has_l:
        return steps, []

    want = 1 if has_r else -1
    first = next((s["joints"][0] for s in steps if s["joints"].get(0)), None)
    if first is None or (first > 0) == (want > 0):
        return steps, []

    lo, hi = JOINT_LIMITS[0]
    flipped = [
        {"joints": {j: (max(lo, min(hi, -a)) if j == 0 else a) for j, a in s["joints"].items()},
         "hold": s["hold"]}
        for s in steps
    ]
    arah = "kanan" if want > 0 else "kiri"
    return flipped, [f"arah dibalik otomatis agar sesuai '{arah}' pada permintaan"]


# ================================================================
# 3. GERAK HALUS (interpolasi)
# ================================================================
def ease_path(current, target, speed=None, hz=None):
    """
    Posisi-posisi antara `current` dan `target` dengan kurva smoothstep (pelan di awal,
    cepat di tengah, pelan di akhir) supaya gerakan tidak kaku/menyentak.
    current/target: {motor: sudut}. Return: list {motor: sudut}, elemen terakhir = target.
    """
    speed = speed or SPEED_DEG_PER_S
    hz = hz or UPDATE_HZ
    start = {j: current.get(j, NEUTRAL_ANGLE) for j in target}
    delta = max(abs(target[j] - start[j]) for j in target)
    if delta == 0:
        return []
    n = max(1, math.ceil(max(MIN_MOVE_S, delta / speed) * hz))

    path, last = [], dict(start)
    for k in range(1, n + 1):
        t = k / n
        e = t * t * (3 - 2 * t)
        pos = {j: int(round(start[j] + (target[j] - start[j]) * e)) for j in target}
        if pos != last:          # lewati posisi yang sama setelah pembulatan
            path.append(pos)
            last = pos
    return path


def _move_to(robot, current, target, dry_run):
    if SMOOTH:
        for pos in ease_path(current, target):
            if not dry_run:
                robot.send_joints(pos)
                time.sleep(1.0 / UPDATE_HZ)
    elif not dry_run:
        robot.send_joints(target)
    current.update(target)


# ================================================================
# 4. MINTA LLM + EKSEKUSI
# ================================================================
def plan_from_request(model, request_text):
    """Panggil LLM sekali (tanpa riwayat chat) dan kembalikan (steps, warnings)."""
    from langchain_core.messages import SystemMessage, HumanMessage
    from llm_utils import strip_thinking, token_counter

    resp = model.invoke([SystemMessage(content=build_motion_prompt()),
                         HumanMessage(content=request_text)])
    token_counter.add(resp)
    raw = resp.content or ""
    if not raw.strip():
        raw = resp.additional_kwargs.get("reasoning_content", "") or ""
    plan = parse_plan(strip_thinking(raw))
    if plan is None:
        return [], ["LLM tidak mengembalikan JSON yang bisa dibaca"]

    steps, warnings = validate_plan(plan)
    steps, dir_warnings = enforce_direction(steps, request_text)
    return steps, warnings + dir_warnings


def execute_plan(robot, steps, dry_run=True):
    """Jalankan langkah-langkah gerakan, lalu SELALU kembalikan motor ke netral."""
    current = {}      # sudut terakhir yang dikirim per motor (anggap mulai dari netral)
    used = set()
    original_error = None
    try:
        for n, step in enumerate(steps, 1):
            target = {j: a * JOINT_SIGN.get(j, 1) for j, a in step["joints"].items()}
            used.update(target)
            print(f"  [Gerakan {n}/{len(steps)}] {target}  tahan {step['hold']:.1f}s"
                  + ("  (DRY-RUN, tidak dikirim)" if dry_run else ""))
            _move_to(robot, current, target, dry_run)
            time.sleep(step["hold"])
    except BaseException as e:
        original_error = e
        raise
    finally:
        # kembali netral -- dijalankan walau ada error/Ctrl+C di tengah
        if used and not dry_run:
            try:
                head_like = {j: NEUTRAL_ANGLE for j in used if j < 8}
                if head_like:
                    _move_to(robot, current, head_like, False)
                    if RELEASE_HEAD_AFTER:
                        time.sleep(HEAD_NEUTRAL_SETTLE_S)
                        robot.send("i")    # lepaskan kontrol manual kepala
                        # BLE write selesai belum berarti firmware telah membaca
                        # perintah. Jangan biarkan 'i\n' bergabung dengan skill
                        # berikutnya di buffer penerima BLE OpenCat.
                        print(f"  [Gerakan] Jeda setelah melepas kontrol kepala: {HEAD_RELEASE_SETTLE_S:g}s")
                        time.sleep(HEAD_RELEASE_SETTLE_S)
                if any(j >= 8 for j in used):
                    robot.send("kup")      # motor kaki: kembali ke pose berdiri
            except Exception as cleanup_error:
                if original_error is None:
                    raise
                print(f"[Gerakan] Gagal kembali netral: {cleanup_error}")
