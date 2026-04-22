import os
from agent_sync.registry import save_to_local_registry


def test_save_to_local_registry(tmp_path):
    registry_dir = tmp_path / "registry"
    os.makedirs(registry_dir)

    # Save first time
    path1 = save_to_local_registry(str(registry_dir), "reviewer", "Body text")
    assert os.path.exists(path1.replace("file://", ""))

    # Save same content, should return same path (deduplication)
    path2 = save_to_local_registry(str(registry_dir), "reviewer_copy", "Body text")
    assert path1 == path2
