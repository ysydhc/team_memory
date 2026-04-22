import yaml
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Dependency:
    name: str
    source: str
    targets: Optional[List[str]] = None
    mode: str = "inline"
    strip_frontmatter: bool = True
    add_warning: bool = True
    commit_sha: Optional[str] = None  # 用于锁定版本 (Lockfile)


@dataclass
class LockfileConfig:
    dependencies: List[Dependency]


def parse_lockfile(path: str) -> LockfileConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    deps = [Dependency(**d) for d in data.get("dependencies", [])]
    return LockfileConfig(dependencies=deps)
