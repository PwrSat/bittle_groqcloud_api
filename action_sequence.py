"""Rangkaian skill bawaan dan gerakan kepala, diperiksa sebelum dikirim."""

import json
import math
import re
import time

import motion_gen
from llm_utils import extract_final_reply

MAX_ACTIONS = 8
DEFAULT_SKILL_SECONDS = 2.0
SKILL_SECONDS_RANGE = (0.5, 10.0)
MAX_SEQUENCE_SECONDS = 60.0  # perkiraan jeda + gerak; tidak termasuk latensi koneksi
GAIT_STOP_SECONDS = 0.5
CONTINUOUS_COMMANDS = {"kwkF", "kbk", "kcrF", "kvtF", "kvtLX", "kphF"}


class ActionPlanError(ValueError):
    pass


def build_system_prompt(commands):
    example = {
        "reply": "Siap, aku duduk, tengok kanan, lalu berdiri!",
        "actions": [
            {"type": "skill", "command": "ksit", "seconds": 2},
            {"type": "motion", "request": "tengok kepala ke kanan"},
            {"type": "skill", "command": "kup", "seconds": 1},
        ],
    }
    return (
        "Kamu adalah Bittle, robot anjing peliharaan yang ramah. Pengguna berbicara "
        "lewat suara. Terjemahkan MAKSUD ucapan menjadi jawaban dan aksi yang berurutan.\n"
        "BAHASA: reply selalu memakai bahasa pengguna, sesuai tag [bahasa: ...]; "
        "jangan tampilkan tag itu. Reply singkat (5-12 kata), kecuali diminta bercerita.\n"
        "FORMAT: balas HANYA satu objek JSON valid dengan dua field: reply (teks) dan "
        "actions (daftar). Jangan gunakan markdown atau format ##command##.\n"
        "AKSI BAWAAN: {\"type\":\"skill\",\"command\":\"ksit\",\"seconds\":2}. "
        "Command harus salah satu KODE dalam daftar berikut, jangan diterjemahkan:\n"
        + json.dumps(commands, ensure_ascii=False)
        + "\nAKSI KEPALA: {\"type\":\"motion\",\"request\":\"tengok kepala ke kanan\"}. "
        "Request hanya potongan permintaan gerakan kepala itu, dalam bahasa pengguna, "
        "bukan seluruh rangkaian. Hanya menoleh kiri-kanan/geleng; tidak mengangguk "
        "atau merancang gerakan kaki baru. Kepala kembali netral setelah tiap aksi motion.\n"
        "URUTAN: untuk permintaan beberapa aksi, tulis SEMUA aksi sesuai urutannya. "
        "Boleh mencampur skill dan motion, atau mengulang aksi dengan menuliskannya lagi. "
        f"Maksimal {MAX_ACTIONS} aksi, perkiraan durasi total maksimal {MAX_SEQUENCE_SECONDS:g} detik. "
        "Jangan menambah gerakan yang tidak diminta. Jika ada gerakan yang tidak didukung "
        "atau rangkaian terlalu panjang, jelaskan singkat dalam reply dan gunakan actions: [].\n"
        f"DURASI: seconds adalah jeda setelah skill dikirim, antara {SKILL_SECONDS_RANGE[0]} "
        f"dan {SKILL_SECONDS_RANGE[1]} detik; default {DEFAULT_SKILL_SECONDS:g}. "
        "Ikuti durasi yang diminta pengguna jika dalam batas. Beri aksi atraksi waktu "
        "yang cukup (misalnya 4 detik), jangan kirim aksi berikutnya langsung. "
        "Berjalan/merangkak/mundur/berputar akan dihentikan otomatis sesudah jedanya.\n"
        "STOP: berhenti/diam/stop berarti skill kup, bukan mundur. Jika ada stop di tengah "
        "rangkaian, letakkan kup pada urutan itu. Patuhi negasi seperti 'jangan duduk'. "
        "Jika hanya mengobrol, actions boleh kosong; untuk sapaan boleh satu skill khi.\n"
        "CONTOH 'duduk, tengok kanan, lalu berdiri':\n"
        + json.dumps(example, ensure_ascii=False)
    )


def _seconds(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActionPlanError(f"{label} harus angka")
    result = float(value)
    if not math.isfinite(result):
        raise ActionPlanError(f"{label} harus angka finite")
    return result


def validate_actions(actions, valid_commands):
    if not isinstance(actions, list) or len(actions) > MAX_ACTIONS:
        raise ActionPlanError(f"actions harus berupa daftar maksimal {MAX_ACTIONS} aksi")
    clean = []
    for index, action in enumerate(actions, 1):
        if not isinstance(action, dict):
            raise ActionPlanError(f"aksi {index}: format salah")
        kind = action.get("type")
        if kind == "skill":
            if set(action) - {"type", "command", "seconds"}:
                raise ActionPlanError(f"aksi {index}: field skill tidak dikenal")
            command = action.get("command")
            if not isinstance(command, str) or command not in valid_commands:
                raise ActionPlanError(f"aksi {index}: kode skill tidak valid: {command!r}")
            seconds = _seconds(action.get("seconds", DEFAULT_SKILL_SECONDS), f"aksi {index}: seconds")
            if not SKILL_SECONDS_RANGE[0] <= seconds <= SKILL_SECONDS_RANGE[1]:
                raise ActionPlanError(f"aksi {index}: seconds di luar batas {SKILL_SECONDS_RANGE}")
            clean.append({"type": "skill", "command": command, "seconds": seconds})
        elif kind == "motion":
            if set(action) - {"type", "request"}:
                raise ActionPlanError(f"aksi {index}: field motion tidak dikenal")
            request = action.get("request")
            if not isinstance(request, str) or not request.strip() or len(request) > 300:
                raise ActionPlanError(f"aksi {index}: permintaan kepala kosong/terlalu panjang")
            clean.append({"type": "motion", "request": request.strip()})
        else:
            raise ActionPlanError(f"aksi {index}: tipe aksi tidak dikenal: {kind!r}")
    duration = sum(a.get("seconds", 0) for a in clean)
    if duration > MAX_SEQUENCE_SECONDS:
        raise ActionPlanError(f"perkiraan rangkaian melebihi {MAX_SEQUENCE_SECONDS:g} detik")
    return clean


def parse_reply(text, valid_commands, request_text=""):
    """Baca format baru; satu command format lama tetap didukung."""
    text = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    if text.startswith(("{", "[")):
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ActionPlanError("jawaban AI bukan JSON valid") from exc
        if not isinstance(payload, dict) or set(payload) != {"reply", "actions"}:
            raise ActionPlanError("jawaban harus punya field reply dan actions")
        reply = payload["reply"]
        if not isinstance(reply, str) or not reply.strip():
            raise ActionPlanError("teks reply kosong/tidak valid")
        return reply.strip(), validate_actions(payload["actions"], valid_commands)

    matches = list(re.finditer(r"##(.*?)##", text))
    if "##" in text and (len(matches) != 1 or text[matches[0].end():].strip()):
        raise ActionPlanError("gunakan JSON actions untuk lebih dari satu gerakan")
    reply, command = extract_final_reply(text)
    if matches and not reply:
        raise ActionPlanError("teks jawaban kosong")
    if command == "motion":
        actions = [{"type": "motion", "request": request_text}]
    elif command:
        actions = [{"type": "skill", "command": command}]
    else:
        actions = []
    return reply, validate_actions(actions, valid_commands)


def _checked_motion_steps(steps):
    if not isinstance(steps, list) or not steps or len(steps) > motion_gen.MAX_STEPS:
        raise ActionPlanError("gerakan kepala tidak memiliki langkah valid")
    for step in steps:
        if not isinstance(step, dict) or not isinstance(step.get("joints"), dict):
            raise ActionPlanError("format langkah kepala tidak valid")
        for angle in step["joints"].values():
            _seconds(angle, "sudut kepala")
        hold = _seconds(step.get("hold"), "waktu tahan kepala")
        if not motion_gen.HOLD_RANGE[0] <= hold <= motion_gen.HOLD_RANGE[1]:
            raise ActionPlanError("waktu tahan kepala di luar batas")
    clean, warnings = motion_gen.validate_plan({"steps": steps})
    if warnings or len(clean) != len(steps):
        raise ActionPlanError("langkah kepala di luar batas yang diizinkan")
    return clean


def _motion_seconds(steps):
    current = {}
    seconds = 0.0
    for step in steps:
        target = step["joints"]
        if motion_gen.SMOOTH:
            seconds += len(motion_gen.ease_path(current, target)) / motion_gen.UPDATE_HZ
        seconds += step["hold"]
        current.update(target)
    if motion_gen.SMOOTH and current:
        neutral = {j: motion_gen.NEUTRAL_ANGLE for j in current}
        seconds += len(motion_gen.ease_path(current, neutral)) / motion_gen.UPDATE_HZ
    if motion_gen.RELEASE_HEAD_AFTER:
        seconds += motion_gen.HEAD_NEUTRAL_SETTLE_S + motion_gen.HEAD_RELEASE_SETTLE_S
    return seconds


def _checked_prepared(actions, valid_commands):
    if not isinstance(actions, list):
        raise ActionPlanError("rangkaian belum disiapkan")
    base = []
    for action in actions:
        if not isinstance(action, dict):
            raise ActionPlanError("format aksi salah")
        stripped = dict(action)
        if action.get("type") == "motion":
            stripped.pop("steps", None)
        base.append(stripped)
    clean = validate_actions(base, valid_commands)
    total = 0.0
    for source, action in zip(actions, clean):
        if action["type"] == "motion":
            action["steps"] = _checked_motion_steps(source.get("steps"))
            total += _motion_seconds(action["steps"])
        else:
            total += action["seconds"]
            if action["command"] in CONTINUOUS_COMMANDS:
                total += GAIT_STOP_SECONDS
    if total > MAX_SEQUENCE_SECONDS:
        raise ActionPlanError(f"perkiraan rangkaian melebihi {MAX_SEQUENCE_SECONDS:g} detik")
    return clean


def prepare_actions(model, actions, valid_commands):
    """Siapkan SEMUA rencana kepala sebelum gerakan pertama; tidak memakai robot."""
    clean = validate_actions(actions, valid_commands)
    warnings = []
    for index, action in enumerate(clean, 1):
        if action["type"] == "motion":
            print(f"[Rencana] Menyiapkan gerakan kepala untuk aksi {index}: {action['request']}")
            steps, notes = motion_gen.plan_from_request(model, action["request"])
            action["steps"] = _checked_motion_steps(steps)
            warnings.extend(f"aksi {index}: {note}" for note in notes)
    return _checked_prepared(clean, valid_commands), warnings


def _send(robot, command):
    result = robot.send(command)
    if result is False or result == -1:
        raise RuntimeError(f"pengiriman {command!r} gagal")


def execute_actions(robot, actions, valid_commands, dry_run=False):
    """Eksekusi bergantian; jeda bukan konfirmasi gerakan dari firmware BLE."""
    clean = _checked_prepared(actions, valid_commands)
    try:
        for index, action in enumerate(clean, 1):
            if action["type"] == "skill":
                command, seconds = action["command"], action["seconds"]
                print(f"[Aksi {index}/{len(clean)}] {command}, jeda {seconds:g}s"
                      + (" (pratinjau)" if dry_run else ""))
                if not dry_run:
                    _send(robot, command)
                    time.sleep(seconds)
                if command in CONTINUOUS_COMMANDS:
                    print("  [Aksi] Menghentikan gerakan berulang: kup")
                    if not dry_run:
                        _send(robot, "kup")
                        time.sleep(GAIT_STOP_SECONDS)
            else:
                print(f"[Aksi {index}/{len(clean)}] kepala: {action['request']}"
                      + (" (pratinjau)" if dry_run else ""))
                if dry_run:
                    for step in action["steps"]:
                        print(f"  motor={step['joints']}, tahan={step['hold']:g}s")
                else:
                    motion_gen.execute_plan(robot, action["steps"], dry_run=False)
    except BaseException:
        # Jangan jalankan sisa antrean atau menutupi error asli jika stop gagal.
        if not dry_run:
            try:
                _send(robot, "kup")
            except Exception as stop_error:
                print(f"[Aksi] Gagal mengirim stop: {stop_error}")
        raise
