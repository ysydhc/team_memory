import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class Dependency:
    name: str
    source: str
    targets: Optional[List[str]] = None
    mode: str = "inline"
    strip_frontmatter: bool = True
    add_warning: bool = True
    commit_sha: Optional[str] = None  # 用于锁定版本 (Lockfile)
    variables: Dict[str, str] = field(default_factory=dict)  # Skill 级别的变量注入


@dataclass
class LockfileConfig:
    target_path: Optional[str] = None
    variables: Dict[str, str] = field(default_factory=dict)  # 项目级别的全局变量
    dependencies: List[Dependency] = field(default_factory=list)


def parse_lockfile(path: str) -> LockfileConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    deps = [Dependency(**d) for d in data.get("dependencies", [])]
    return LockfileConfig(
        target_path=data.get("target_path"),
        variables=data.get("variables", {}),
        dependencies=deps,
    )
