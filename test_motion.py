"""
Tes fitur gerakan motor. Urutan yang disarankan:

  python test_motion.py --selftest                # 1. tes validator, TANPA internet/robot
  python test_motion.py --dry "gelengkan kepala"  # 2. LLM menyusun gerakan, cuma DITAMPILKAN
  python test_motion.py --run "gelengkan kepala"  # 3. benar-benar menggerakkan robot (BLE)
"""
import sys


def selftest():
    import motion_gen as m

    # 1. JSON dibungkus markdown tetap terbaca
    plan = m.parse_plan('```json\n{"steps":[{"joints":{"0":40},"hold":0.5}]}\n```')
    steps, warn = m.validate_plan(plan)
    assert steps == [{"joints": {0: 40}, "hold": 0.5}] and not warn, (steps, warn)

    # 2. sudut berlebihan dipotong ke batas aman
    steps, warn = m.validate_plan({"steps": [{"joints": {"0": 999}, "hold": 1}]})
    assert steps[0]["joints"][0] == m.JOINT_LIMITS[0][1] and warn

    # 3. motor yang tidak diizinkan (misal kaki 12) dibuang total
    steps, warn = m.validate_plan({"steps": [{"joints": {"12": 30}, "hold": 1}]})
    assert steps == [] and warn

    # 4. langkah & durasi dibatasi
    many = {"steps": [{"joints": {"0": 10}, "hold": 2.0}] * 20}
    steps, _ = m.validate_plan(many)
    assert len(steps) <= m.MAX_STEPS
    assert sum(s["hold"] for s in steps) <= m.MAX_TOTAL_SECONDS

    # 5. sampah dari LLM tidak bikin crash
    assert m.parse_plan("maaf saya tidak bisa") is None
    assert m.validate_plan("bukan dict") == ([], ["rencana tidak punya daftar 'steps'"])

    # 6. arah kiri/kanan dipaksa konsisten dengan kata di permintaan
    st = [{"joints": {0: -30}, "hold": 0.5}]                       # LLM kasih negatif (= kiri)
    fixed, w = m.enforce_direction(st, "tengok kanan")
    assert fixed[0]["joints"][0] == 30 and w                       # harus jadi positif (= kanan)
    same, w = m.enforce_direction(st, "tengok kiri")
    assert same[0]["joints"][0] == -30 and not w                   # sudah benar -> tidak diubah
    both, w = m.enforce_direction(st, "celingukan kiri kanan")
    assert both == st and not w                                    # dua arah -> dibiarkan

    # 7. interpolasi halus: banyak langkah kecil, berakhir tepat di target, tidak mundur
    path = m.ease_path({0: 0}, {0: 40})
    angles = [p[0] for p in path]
    assert len(angles) > 3 and angles[-1] == 40, angles
    assert angles == sorted(angles), angles
    assert m.ease_path({0: 40}, {0: 40}) == []

    print("Semua tes validator LULUS.")


def make_model():
    from dotenv import load_dotenv
    from llm_utils import get_model
    load_dotenv()
    return get_model("groq")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "--selftest":
        selftest()
    elif args[0] in ("--dry", "--run") and len(args) >= 2:
        import motion_gen as m
        request = " ".join(args[1:])
        steps, warnings = m.plan_from_request(make_model(), request)
        for w in warnings:
            print("[Validasi]", w)
        if not steps:
            print("Tidak ada langkah valid, tidak ada yang dijalankan.")
            sys.exit(1)
        if args[0] == "--dry":
            m.execute_plan(None, steps, dry_run=True)
        else:
            from robot_link import create_robot
            robot = create_robot("ble")
            try:
                m.execute_plan(robot, steps, dry_run=False)
            finally:
                robot.close()
    else:
        print(__doc__)