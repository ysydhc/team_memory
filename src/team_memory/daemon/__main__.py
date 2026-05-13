"""TM Daemon entry point.

Run with: python -m team_memory.daemon
"""

import logging

import uvicorn

from team_memory.daemon.runtime.app import create_app
from team_memory.daemon.runtime.config import load_config


def main() -> None:
    # Configure app-level logging so our [READ]/[WRITE]/[WATCH] lines are visible
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config()
    app = create_app(config)
    uvicorn.run(
        app,
        host=config.daemon.host,
        port=config.daemon.port,
        log_level="info",
        # Preserve app loggers (don't let uvicorn override them)
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": "uvicorn",
                    "stream": "ext://sys.stdout",
                },
            },
            "formatters": {
                "uvicorn": {
                    "format": "%(asctime)s %(levelname)s %(message)s",
                    "datefmt": "%H:%M:%S",
                },
            },
            "loggers": {
                "daemon.app": {"handlers": ["default"], "level": "INFO", "propagate": False},
                "tm_daemon.watcher": {"handlers": ["default"], "level": "INFO", "propagate": False},
                "tm_daemon.pipeline": {"handlers": ["default"], "level": "INFO", "propagate": False},
            },
        },
    )


if __name__ == "__main__":
    main()
