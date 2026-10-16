"""
Test koneksi Bluetooth ke Bittle, TANPA LLM/suara.

    python tools/check_ble.py --scan      # lihat semua device BLE di sekitar
    python tools/check_ble.py             # connect, duduk, berdiri, putus
"""
import sys
import time
import asyncio
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


async def scan():
    from bleak import BleakScanner
    print("Scan 8 detik...\n")
    found = await BleakScanner.discover(timeout=8, return_adv=True)
    for addr, (dev, adv) in sorted(found.items(), key=lambda kv: -(kv[1][1].rssi or -999)):
        print(f"{(dev.name or adv.local_name or '(tanpa nama)'):<28} {addr}  RSSI={adv.rssi}")


if __name__ == "__main__":
    if "--scan" in sys.argv:
        asyncio.run(scan())
    else:
        from robot_link import create_robot
        robot = create_robot("ble")
        try:
            print("Kirim: ksit");  robot.send("ksit"); time.sleep(3)
            print("Kirim: kup");   robot.send("kup");  time.sleep(3)
            print("Berhasil.")
        finally:
            robot.close()
