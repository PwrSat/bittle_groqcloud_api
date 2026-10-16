
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    from ardSerial import connectPort, send, closeAllSerial, logger

    goodPorts = {}
    try:
        connectPort(goodPorts)
        if len(goodPorts) > 0:
            time.sleep(3);
            send(goodPorts, ['g', 1])
            time.sleep(1);
            send(goodPorts, ['ksit', 1])
            time.sleep(1);
            send(goodPorts, ['krest', 1])
            time.sleep(1);
            send(goodPorts, ['kzero', 1])
            closeAllSerial(goodPorts)
            logger.info("finish!")

    except Exception as e:
        logger.info("Exception")
        closeAllSerial(goodPorts)
        raise e


if __name__ == '__main__':
    main()
