"""Uji duduk melalui USB; hanya bergerak saat dijalankan langsung."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    from ardSerial import connectPort, send, closeAllSerial

    ports = {}
    try:
        connectPort(ports)
        if not ports:
            raise RuntimeError("Robot USB tidak ditemukan.")
        send(ports, ["ksit", 1])
    finally:
        closeAllSerial(ports)


if __name__ == "__main__":
    main()
