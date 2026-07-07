import platform
import socket
import sys
from datetime import datetime


def main() -> None:
    print("Raspberry Pi deployment test successful.")
    print(f"Time: {datetime.now().isoformat(timespec='seconds')}")
    print(f"Hostname: {socket.gethostname()}")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")


if __name__ == "__main__":
    main()
