"""TM Daemon entry point.

Run with: python -m daemon
"""

import uvicorn

from daemon.app import create_app
from daemon.config import load_config


def main() -> None:
    config = load_config()
    app = create_app(config)
    uvicorn.run(app, host=config.daemon.host, port=config.daemon.port)


if __name__ == "__main__":
    main()
