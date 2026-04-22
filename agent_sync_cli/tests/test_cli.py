import subprocess
import sys


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "agent_sync.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert "AgentSync" in result.stdout
    assert "install" in result.stdout
