"""Peluncur kompatibilitas; alat aslinya ada di tools/check_motion.py."""

import runpy


if __name__ == "__main__":
    runpy.run_module("tools.check_motion", run_name="__main__")
