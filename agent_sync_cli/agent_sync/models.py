import yaml
from dataclasses import dataclass
from typing import List


@dataclass
class Dependency:
    name: str
    source: str


@dataclass
class LockfileConfig:
    dependencies: List[Dependency]


def parse_lockfile(path: str) -> LockfileConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    deps = [Dependency(**d) for d in data.get("dependencies", [])]
    return LockfileConfig(dependencies=deps)
