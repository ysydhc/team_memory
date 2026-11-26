"""Unit tests for Git Worktree project name resolution in project_resolver."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from team_memory.project_resolver import (
    _git_repo_name,
    _repo_name_from_remote_url,
    infer_project_name,
)


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo_with_commit(repo_dir: Path) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    _git(["init"], repo_dir)
    _git(["config", "user.email", "test@example.com"], repo_dir)
    _git(["config", "user.name", "Test User"], repo_dir)
    (repo_dir / "README.md").write_text("hello\n", encoding="utf-8")
    _git(["add", "README.md"], repo_dir)
    _git(["commit", "-m", "initial"], repo_dir)


@pytest.mark.unit
class TestGitWorktreeProjectResolution:
    def test_worktree_resolves_to_main_repo_name(self, tmp_path: Path) -> None:
        """Main repo and git worktree (plus nested subdirs) must infer identical project name."""
        main_repo = tmp_path / "adx-sdk-android"
        _init_repo_with_commit(main_repo)

        nested_main = main_repo / "app" / "src"
        nested_main.mkdir(parents=True)

        worktree_dir = tmp_path / "android_llog_post_cr_ae0b35d25bd3"
        _git(["worktree", "add", "-b", "feature/llog-fix", str(worktree_dir)], main_repo)

        nested_wt = worktree_dir / "app" / "src"
        nested_wt.mkdir(parents=True, exist_ok=True)

        assert infer_project_name(str(main_repo)) == "adx_sdk_android"
        assert infer_project_name(str(nested_main)) == "adx_sdk_android"
        assert infer_project_name(str(worktree_dir)) == "adx_sdk_android"
        assert infer_project_name(str(nested_wt)) == "adx_sdk_android"

    def test_tm_toml_overrides_worktree_git_name(self, tmp_path: Path) -> None:
        """Priority chain: .tm.toml explicit name > git main repo name > dir basename."""
        main_repo = tmp_path / "adx-sdk-android"
        _init_repo_with_commit(main_repo)

        worktree_dir = tmp_path / "wt_override_1234567890ab"
        _git(["worktree", "add", "-b", "feature/override", str(worktree_dir)], main_repo)
        (worktree_dir / ".tm.toml").write_text('[project]\nname = "custom-override"\n', encoding="utf-8")

        assert infer_project_name(str(worktree_dir)) == "custom_override"
        assert infer_project_name(str(main_repo)) == "adx_sdk_android"

    def test_bare_repo_falls_back_to_origin_remote_or_dir_name(self, tmp_path: Path) -> None:
        """Bare repo resolves from origin remote URL, or strips .git from bare dir name."""
        bare_dir = tmp_path / "my-bare-service.git"
        bare_dir.mkdir()
        _git(["init", "--bare"], bare_dir)

        # Without origin remote: strips .git from bare directory name
        assert infer_project_name(str(bare_dir)) == "my_bare_service"

        # With origin remote: prefers origin repo name
        _git(["remote", "add", "origin", "git@icode.baidu.com:baidu/mobile/adx-sdk-ios.git"], bare_dir)
        assert infer_project_name(str(bare_dir)) == "adx_sdk_ios"

    def test_older_git_without_path_format_absolute_fallback(self, tmp_path: Path) -> None:
        """Simulate older git where --path-format=absolute fails and relative --git-common-dir is used."""
        main_repo = tmp_path / "adx-sdk-ios"
        _init_repo_with_commit(main_repo)
        worktree_dir = tmp_path / "ios_exp_click_result_653b785dc360"
        _git(["worktree", "add", "-b", "feature/ios-exp", str(worktree_dir)], main_repo)

        real_run_git = __import__("team_memory.project_resolver", fromlist=["_run_git"])._run_git

        def fake_run_git(args: list[str], cwd: str) -> str | None:
            if "--path-format=absolute" in args:
                return None
            return real_run_git(args, cwd)

        with patch("team_memory.project_resolver._run_git", side_effect=fake_run_git):
            assert _git_repo_name(str(worktree_dir)) == "adx-sdk-ios"
            assert infer_project_name(str(worktree_dir)) == "adx_sdk_ios"

    def test_non_git_directory_falls_back_to_basename(self, tmp_path: Path) -> None:
        plain_dir = tmp_path / "Plain-Project"
        plain_dir.mkdir()
        assert _git_repo_name(str(plain_dir)) is None
        assert infer_project_name(str(plain_dir)) == "plain_project"

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://github.com/org/adx-sdk-android.git", "adx-sdk-android"),
            ("https://github.com/org/adx-sdk-android/", "adx-sdk-android"),
            ("git@icode.baidu.com:baidu/gensoft-ipcs-mobile/adx-sdk-ios.git", "adx-sdk-ios"),
            ("ssh://git@host:2222/team/team_doc.git", "team_doc"),
            ("/local/path/to/inapp-adx.git", "inapp-adx"),
            ("", None),
        ],
    )
    def test_repo_name_from_remote_url(self, url: str, expected: str | None) -> None:
        assert _repo_name_from_remote_url(url) == expected
