import argparse


def main():
    parser = argparse.ArgumentParser(description="AgentSync: AI Agent 包管理器")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("install", help="从 lockfile 安装 agents")
    subparsers.add_parser("import", help="导入遗留的 agents")
    args = parser.parse_args()


if __name__ == "__main__":
    main()
