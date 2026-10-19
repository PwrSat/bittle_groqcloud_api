"""Run the Bittle application with the explicit cached medium GPU profile."""

from pathlib import Path
import runpy
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    import speechtotext

    speechtotext.configure_transcriber(model="medium", device="cuda", compute_type="int8_float16")
    runpy.run_path(str(PROJECT_ROOT / "BittleLLM.py"), run_name="__main__")


if __name__ == "__main__":
    main()
