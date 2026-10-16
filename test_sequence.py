"""Peluncur kompatibilitas; alat aslinya ada di tools/check_sequence.py."""

import runpy


if __name__ == "__main__":
    runpy.run_module("tools.check_sequence", run_name="__main__")
