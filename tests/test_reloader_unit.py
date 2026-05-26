"""Unit tests for reloader.py — lightweight daemon auto-reloader.

Covers:
  - get_mtime_map()     pure file-system logic (tempdir)
  - restart_daemon()    subprocess calls (mocked)
  - main()              argument parsing + watch loop (mocked sleep/time)

Marked:
  @pytest.mark.unit — no real DB, no real launchctl, mock-only
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
from team_memory.daemon.runtime.reloader import (
    DAEMON_DIR,
    DAEMON_PLIST,
    DEFAULT_INTERVAL,
    get_mtime_map,
    main,
    restart_daemon,
)

pytestmark = pytest.mark.unit


# ===========================================================================
# Module-level constants
# ===========================================================================


class TestModuleConstants:
    """Verify module-level constants are set correctly."""

    def test_daemon_plist_name(self):
        assert DAEMON_PLIST == "com.teammemory.daemon"

    def test_default_interval(self):
        assert DEFAULT_INTERVAL == 3
        assert isinstance(DEFAULT_INTERVAL, int)

    def test_daemon_dir_is_path(self):
        assert isinstance(DAEMON_DIR, Path)

    def test_daemon_dir_ends_with_daemon(self):
        assert str(DAEMON_DIR).endswith("daemon")


# ===========================================================================
# get_mtime_map — file-system logic with tempdir
# ===========================================================================


class TestGetMtimeMap:
    """get_mtime_map() reads .py file modification times."""

    def test_empty_directory(self, tmp_path):
        """Empty directory should return empty dict."""
        result = get_mtime_map(tmp_path)
        assert result == {}

    def test_ignores_non_py_files(self, tmp_path):
        """Non-.py files should be ignored."""
        (tmp_path / "readme.txt").write_text("hello")
        (tmp_path / "data.json").write_text("{}")
        result = get_mtime_map(tmp_path)
        assert result == {}

    def test_finds_py_files(self, tmp_path):
        """Should find .py files."""
        (tmp_path / "foo.py").write_text("x = 1")
        (tmp_path / "bar.py").write_text("y = 2")
        result = get_mtime_map(tmp_path)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result.values())

    def test_keys_are_string_paths(self, tmp_path):
        """Keys should be string representations of file paths."""
        (tmp_path / "test.py").write_text("pass")
        result = get_mtime_map(tmp_path)
        keys = list(result.keys())
        assert len(keys) == 1
        assert isinstance(keys[0], str)
        assert keys[0].endswith("test.py")

    def test_values_are_timestamps(self, tmp_path):
        """Values should be positive timestamps (mtime)."""
        f = tmp_path / "mod.py"
        f.write_text("pass")
        result = get_mtime_map(tmp_path)
        mtime = list(result.values())[0]
        assert mtime > 0

    def test_directories_are_excluded(self, tmp_path):
        """Subdirectories should not appear."""
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "nested.py").write_text("pass")
        (tmp_path / "top.py").write_text("pass")
        result = get_mtime_map(tmp_path)
        assert len(result) == 1

    def test_multiple_py_files(self, tmp_path):
        """Multiple .py files in the directory."""
        for name in ["a.py", "b.py", "c.py", "d.py"]:
            (tmp_path / name).write_text(f"# {name}")
        result = get_mtime_map(tmp_path)
        assert len(result) == 4


# ===========================================================================
# restart_daemon — subprocess calls (mocked)
# ===========================================================================


class TestRestartDaemon:
    """restart_daemon() calls launchctl unload/load."""

    @patch("team_memory.daemon.runtime.reloader.time.sleep")
    @patch("team_memory.daemon.runtime.reloader.subprocess.run")
    def test_success_returns_true(self, mock_run, mock_sleep):
        """Successful restart should return True."""
        mock_run.return_value = SimpleNamespace(returncode=0)
        result = restart_daemon()
        assert result is True
        assert mock_run.call_count == 2

    @patch("team_memory.daemon.runtime.reloader.time.sleep")
    @patch("team_memory.daemon.runtime.reloader.subprocess.run")
    def test_failure_returns_false(self, mock_run, mock_sleep):
        """Failed load should return False."""
        # First call (unload) succeeds, second (load) fails
        mock_run.side_effect = [
            SimpleNamespace(returncode=0),
            SimpleNamespace(returncode=1, stderr=b"load failed"),
        ]
        result = restart_daemon()
        assert result is False

    @patch("team_memory.daemon.runtime.reloader.time.sleep")
    @patch("team_memory.daemon.runtime.reloader.subprocess.run", side_effect=Exception("launchctl not found"))
    def test_exception_returns_false(self, mock_run, mock_sleep):
        """Exception should be caught, returning False."""
        result = restart_daemon()
        assert result is False

    @patch("team_memory.daemon.runtime.reloader.time.sleep")
    @patch("team_memory.daemon.runtime.reloader.subprocess.run")
    def test_unload_called_first(self, mock_run, mock_sleep):
        """Unload should be called before load."""
        mock_run.return_value = SimpleNamespace(returncode=0)
        restart_daemon()
        first_call = mock_run.call_args_list[0]
        assert "unload" in first_call[0][0]
        second_call = mock_run.call_args_list[1]
        assert "load" in second_call[0][0]

    @patch("team_memory.daemon.runtime.reloader.time.sleep")
    @patch("team_memory.daemon.runtime.reloader.subprocess.run")
    def test_plist_path_uses_home(self, mock_run, mock_sleep):
        """Plist path should be under ~/Library/LaunchAgents."""
        mock_run.return_value = SimpleNamespace(returncode=0)
        restart_daemon()
        for c in mock_run.call_args_list:
            plist_arg = c[0][0][-1]  # last arg of launchctl command
            assert "Library/LaunchAgents" in plist_arg
            assert DAEMON_PLIST in plist_arg

    @patch("team_memory.daemon.runtime.reloader.time.sleep")
    @patch("team_memory.daemon.runtime.reloader.subprocess.run")
    def test_sleep_between_unload_and_load(self, mock_run, mock_sleep):
        """Should sleep 1s between unload and load."""
        mock_run.return_value = SimpleNamespace(returncode=0)
        restart_daemon()
        mock_sleep.assert_called_with(1)


# ===========================================================================
# main() — argument parsing + watch loop (mocked)
# ===========================================================================


class TestMainArgparse:
    """main() parses sys.argv for interval."""

    @patch("team_memory.daemon.runtime.reloader.restart_daemon")
    @patch("team_memory.daemon.runtime.reloader.time.sleep")
    @patch("team_memory.daemon.runtime.reloader.get_mtime_map", return_value={"a.py": 1.0})
    def test_default_interval(self, mock_mtime, mock_sleep, mock_restart):
        """No argv => default interval of 3."""
        with patch.object(sys, "argv", ["reloader.py"]):
            # Break the infinite loop after one iteration
            call_count = 0

            def fake_sleep(secs):
                nonlocal call_count
                call_count += 1
                if call_count > 1:
                    raise KeyboardInterrupt()

            mock_sleep.side_effect = fake_sleep

            with pytest.raises(KeyboardInterrupt):
                main()

    @patch("team_memory.daemon.runtime.reloader.restart_daemon")
    @patch("team_memory.daemon.runtime.reloader.time.sleep")
    @patch("team_memory.daemon.runtime.reloader.get_mtime_map", return_value={"a.py": 1.0})
    def test_custom_interval(self, mock_mtime, mock_sleep, mock_restart):
        """argv[1] sets the interval."""
        with patch.object(sys, "argv", ["reloader.py", "5"]):
            call_count = 0

            def fake_sleep(secs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    assert secs == 5
                if call_count > 1:
                    raise KeyboardInterrupt()

            mock_sleep.side_effect = fake_sleep

            with pytest.raises(KeyboardInterrupt):
                main()

    @patch("team_memory.daemon.runtime.reloader.restart_daemon")
    @patch("team_memory.daemon.runtime.reloader.time.sleep")
    @patch("team_memory.daemon.runtime.reloader.get_mtime_map")
    def test_detects_changes_and_restarts(self, mock_mtime, mock_sleep, mock_restart):
        """When file mtime changes, restart should be called."""
        with patch.object(sys, "argv", ["reloader.py", "1"]):
            call_count = 0

            def fake_sleep(secs):
                nonlocal call_count
                call_count += 1

            mock_sleep.side_effect = fake_sleep

            # First call: baseline, second call: changed file
            mock_mtime.side_effect = [
                {"a.py": 1.0},  # initial
                {"a.py": 2.0},  # changed
                {"a.py": 2.0},  # after restart (break loop)
            ]

            # Make get_mtime_map stop the loop
            loop_count = [0]

            def smart_mtime(d):
                loop_count[0] += 1
                if loop_count[0] == 1:
                    return {"a.py": 1.0}
                elif loop_count[0] == 2:
                    return {"a.py": 2.0}  # changed
                else:
                    raise KeyboardInterrupt()

            mock_mtime.side_effect = smart_mtime

            with pytest.raises(KeyboardInterrupt):
                main()

            mock_restart.assert_called()

    @patch("team_memory.daemon.runtime.reloader.restart_daemon")
    @patch("team_memory.daemon.runtime.reloader.time.sleep")
    @patch("team_memory.daemon.runtime.reloader.get_mtime_map")
    def test_no_change_no_restart(self, mock_mtime, mock_sleep, mock_restart):
        """When nothing changes, restart should NOT be called."""
        with patch.object(sys, "argv", ["reloader.py", "1"]):
            loop_count = [0]

            def smart_sleep(secs):
                pass

            mock_sleep.side_effect = smart_sleep

            def smart_mtime(d):
                loop_count[0] += 1
                if loop_count[0] <= 2:
                    return {"a.py": 1.0}
                raise KeyboardInterrupt()

            mock_mtime.side_effect = smart_mtime

            with pytest.raises(KeyboardInterrupt):
                main()

            mock_restart.assert_not_called()

    @patch("team_memory.daemon.runtime.reloader.restart_daemon")
    @patch("team_memory.daemon.runtime.reloader.time.sleep")
    @patch("team_memory.daemon.runtime.reloader.get_mtime_map")
    def test_new_file_detected(self, mock_mtime, mock_sleep, mock_restart):
        """New .py files appearing should trigger restart."""
        with patch.object(sys, "argv", ["reloader.py", "1"]):
            mock_sleep.side_effect = lambda s: None

            loop_count = [0]

            def smart_mtime(d):
                loop_count[0] += 1
                if loop_count[0] == 1:
                    return {"a.py": 1.0}
                elif loop_count[0] == 2:
                    return {"a.py": 1.0, "b.py": 3.0}  # new file
                raise KeyboardInterrupt()

            mock_mtime.side_effect = smart_mtime

            with pytest.raises(KeyboardInterrupt):
                main()

            mock_restart.assert_called()

    @patch("team_memory.daemon.runtime.reloader.restart_daemon")
    @patch("team_memory.daemon.runtime.reloader.time.sleep")
    @patch("team_memory.daemon.runtime.reloader.get_mtime_map")
    def test_output_on_change(self, mock_mtime, mock_sleep, mock_restart, capsys):
        """Should print detected changes."""
        with patch.object(sys, "argv", ["reloader.py", "1"]):
            mock_sleep.side_effect = lambda s: None

            loop_count = [0]

            def smart_mtime(d):
                loop_count[0] += 1
                if loop_count[0] == 1:
                    return {"a.py": 1.0}
                elif loop_count[0] == 2:
                    return {"a.py": 9.9}
                raise KeyboardInterrupt()

            mock_mtime.side_effect = smart_mtime

            with pytest.raises(KeyboardInterrupt):
                main()

            output = capsys.readouterr().out
            assert "Detected changes" in output

    @patch("team_memory.daemon.runtime.reloader.restart_daemon")
    @patch("team_memory.daemon.runtime.reloader.time.sleep")
    @patch("team_memory.daemon.runtime.reloader.get_mtime_map")
    def test_watching_output(self, mock_mtime, mock_sleep, mock_restart, capsys):
        """Should print watching message on start."""
        mock_mtime.return_value = {}
        mock_sleep.side_effect = KeyboardInterrupt()

        with patch.object(sys, "argv", ["reloader.py", "7"]):
            with pytest.raises(KeyboardInterrupt):
                main()

            output = capsys.readouterr().out
            assert "Watching" in output
            assert "7s" in output
