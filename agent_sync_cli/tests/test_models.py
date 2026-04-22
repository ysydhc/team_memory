from agent_sync.models import parse_lockfile


def test_parse_lockfile(tmp_path):
    lockfile = tmp_path / "agent-lock.yaml"
    lockfile.write_text(
        "dependencies:\n  - name: reviewer\n    source: https://github.com/org/repo/tree/main/skills"
    )

    config = parse_lockfile(str(lockfile))
    assert len(config.dependencies) == 1
    assert config.dependencies[0].name == "reviewer"
