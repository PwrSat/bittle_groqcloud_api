"""
robot_link.py -- lapisan koneksi ke robot Bittle (USB serial ATAU Bluetooth BLE)

Dua class dengan interface SAMA, jadi BittleLLM.py cukup panggil:
    robot = create_robot("ble")      # atau "serial"
    robot.send("kwkF")               # kirim skill
    robot.close()

BLE memakai Nordic UART Service milik BiBoard (ESP32), sesuai panduan Petoi:
    Service : 6E400001-B5A3-F393-E0A9-E50E24DCCA9E
    RX (tulis ke robot)  : 6E400002-B5A3-F393-E0A9-E50E24DCCA9E
    TX (notify dari robot): 6E400003-B5A3-F393-E0A9-E50E24DCCA9E
Format data persis sama dengan serial: teks skill + newline, contoh b"ksit\n".
"""
import asyncio
import os
import threading
import time

NUS_SERVICE = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
NUS_RX = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"   # tulis ke robot
NUS_TX = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"   # notifikasi dari robot


# ---------------------------------------------------------------
# Mode 1: USB serial (cara lama, lewat ardSerial.py)
# ---------------------------------------------------------------
class SerialRobot:
    def __init__(self):
        import ardSerial as _ard  # import di sini supaya mode BLE tidak butuh ardSerial
        self._ard = _ard
        self._ports = {}
        _ard.connectPort(self._ports)
        if not self._ports:
            raise RuntimeError("Robot tidak ditemukan lewat serial/USB.")

    def send(self, skill):
        self._ard.send(self._ports, [skill, 1])

    def send_joints(self, angles):
        """Gerakkan beberapa motor sekaligus. angles = {index_motor: sudut}."""
        flat = []
        for idx, ang in angles.items():
            flat += [int(idx), int(ang)]
        self._ard.send(self._ports, ["i", flat, 0])   # format resmi Petoi: ['i', [idx, sudut, ...], delay]

    def close(self):
        for p in list(self._ports.keys()):
            try:
                p.Close_Engine()
            except Exception:
                pass


# ---------------------------------------------------------------
# Mode 2: Bluetooth Low Energy (pakai library bleak)
# ---------------------------------------------------------------
class BleRobot:
    def __init__(self, address=None, name_prefix="Bittle", scan_time=8.0):
        """
        address     : MAC address BLE robot (opsional, lebih cepat kalau diisi)
        name_prefix : kalau address kosong, cari device yang namanya diawali ini
        """
        # alamat bisa juga diisi lewat .env: BITTLE_BLE_ADDRESS=AA:BB:CC:DD:EE:FF
        self._address = address or os.getenv("BITTLE_BLE_ADDRESS")
        self._name_prefix = name_prefix
        self._scan_time = scan_time
        self._client = None

        # event loop asyncio jalan di thread terpisah, supaya BittleLLM.py
        # yang sinkron (biasa) bisa tinggal panggil robot.send(...)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        self._run(self._connect(), timeout=scan_time * 2 + 20)

    # --- helper ---
    def _run(self, coro, timeout=10):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    async def _find_device(self):
        from bleak import BleakScanner
        seen = []
        for attempt in (1, 2):   # BLE advertising kadang telat, jadi scan 2x
            print(f"[BLE] Scan {self._scan_time:.0f} detik (percobaan {attempt}/2)...")
            found = await BleakScanner.discover(timeout=self._scan_time, return_adv=True)
            for addr, (dev, adv) in found.items():
                name = dev.name or adv.local_name or ""
                seen.append(name or "(tanpa nama)")
                has_nus = NUS_SERVICE.lower() in [u.lower() for u in (adv.service_uuids or [])]
                # abaikan versi "_SSP" (itu Bluetooth classic, bukan BLE)
                if (name.startswith(self._name_prefix) and not name.endswith("_SSP")) or has_nus:
                    print(f"[BLE] Ketemu: {name or '(tanpa nama)'}  [{addr}]")
                    return addr
        print("[BLE] Tidak ada yang cocok. Device yang terlihat: "
              + (", ".join(sorted(set(seen))[:15]) or "(tidak ada sama sekali)"))
        return None

    async def _connect(self):
        from bleak import BleakClient
        addr = self._address or await self._find_device()
        if not addr:
            raise RuntimeError(
                "Robot BLE tidak ketemu. Pastikan robot nyala, jarak dekat, "
                "tidak sedang tersambung ke HP/aplikasi Petoi, dan Bluetooth "
                "laptop aktif. Jalankan test_ble.py --scan buat lihat daftar device."
            )
        self._address = addr
        self._client = BleakClient(addr)
        await self._client.connect()
        print(f"[BLE] Terhubung ke {addr}")

    async def _write(self, data: bytes):
        # BLE default cuma muat ~20 byte per paket -> potong
        for i in range(0, len(data), 20):
            await self._client.write_gatt_char(NUS_RX, data[i:i + 20], response=False)
            await asyncio.sleep(0.01)

    # --- API publik ---
    def send(self, skill):
        data = (skill + "\n").encode("utf-8")
        try:
            if not (self._client and self._client.is_connected):
                print("[BLE] Koneksi putus, mencoba menyambung ulang...")
                self._run(self._connect(), timeout=30)
            self._run(self._write(data), timeout=5)
        except Exception as e:
            print(f"[BLE] Gagal kirim '{skill}': {e}")

    def send_joints(self, angles):
        """Gerakkan beberapa motor sekaligus. angles = {index_motor: sudut}.
        Dikirim sebagai teks 'i 0 40 8 -20' (protokol serial Petoi)."""
        parts = []
        for idx, ang in angles.items():
            parts += [str(int(idx)), str(int(ang))]
        self.send("i " + " ".join(parts))

    def close(self):
        try:
            if self._client and self._client.is_connected:
                self._run(self._client.disconnect(), timeout=5)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)


def create_robot(mode="ble", **kwargs):
    """mode: 'ble' (Bluetooth) atau 'serial' (USB)."""
    if mode == "ble":
        return BleRobot(**kwargs)
    if mode == "serial":
        return SerialRobot()
    raise ValueError("mode harus 'ble' atau 'serial'")