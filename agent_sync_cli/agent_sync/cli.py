import argparse


def main():
    parser = argparse.ArgumentParser(description="AgentSync: AI Agent 包管理器")
    subparsers = parser.add_subparsers(dest="command")

    # 核心生命周期命令
    subparsers.add_parser("install", help="从配置中心安装 agents 到目标项目")
    subparsers.add_parser("update", help="强制更新远程源并重新生成 agents")
    subparsers.add_parser("status", help="检查配置漂移 (Drift Detection) 与同步状态")
    subparsers.add_parser("import", help="导入遗留的 agents 老项目")
    subparsers.add_parser("push", help="将本地修改的 skill 反推回中央仓库")

    # 极简交互与发现命令 (UX 优化)
    subparsers.add_parser("link", help="在当前业务项目目录执行，自动将其绑定到配置中心")
    subparsers.add_parser("add", help="向当前绑定的项目添加新的 Skill 链接")
    subparsers.add_parser("search", help="浏览远程 GitHub 目录中的可用 Skills")
    subparsers.add_parser(
        "check", help="CI/CD 门禁：检查配置与实际文件是否一致，不一致则报错"
    )

    args = parser.parse_args()


if __name__ == "__main__":
    main()
