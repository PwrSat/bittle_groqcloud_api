"""Uji rangkaian lewat teks, memakai planner yang sama dengan BittleLLM.py.

python tools/check_sequence.py --dry "duduk, tengok kanan, lalu berdiri"
python tools/check_sequence.py --run "jalan tiga detik, lalu duduk"
"""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import action_sequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry", action="store_true", help="pratinjau tanpa robot (default)")
    mode.add_argument("--run", action="store_true", help="kirim rangkaian ke robot")
    parser.add_argument("--connection", choices=("ble", "serial"), default="ble")
    parser.add_argument("request", help="permintaan gerakan dalam bahasa sehari-hari")
    args = parser.parse_args()

    from dotenv import load_dotenv
    from langchain_core.messages import SystemMessage, HumanMessage
    from llm_utils import get_model, strip_thinking, token_counter

    load_dotenv(PROJECT_ROOT / ".env")
    commands = json.loads((PROJECT_ROOT / "Commands.json").read_text(encoding="utf-8"))
    valid_commands = set(commands.values())
    model = get_model("groq")
    response = model.invoke([
        SystemMessage(content=action_sequence.build_system_prompt(commands)),
        HumanMessage(content=args.request),
    ])
    token_counter.add(response)
    raw = response.content or response.additional_kwargs.get("reasoning_content", "")
    reply, actions = action_sequence.parse_reply(
        strip_thinking(raw), valid_commands, request_text=args.request
    )
    prepared, warnings = action_sequence.prepare_actions(model, actions, valid_commands)
    print(f"[Bittle]: {reply}")
    for warning in warnings:
        print(f"[Validasi] {warning}")

    robot = None
    try:
        if args.run and prepared:
            from robot_link import create_robot

            robot = create_robot(args.connection)
        action_sequence.execute_actions(robot, prepared, valid_commands, dry_run=not args.run)
    finally:
        if robot is not None:
            robot.close()
        print(f"[Token] Total: {token_counter.total}, {token_counter.calls} panggilan")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nRangkaian dibatalkan.")
    except Exception as error:
        raise SystemExit(f"Rangkaian gagal: {error}")
