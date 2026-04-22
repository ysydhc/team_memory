import re


def parse_github_url(url: str, use_ssh: bool = False):
    match = re.match(r"(https://github\.com/[^/]+/[^/]+)/tree/([^/]+)/(.*)", url)
    if match:
        repo_url = match.group(1) + ".git"
        if use_ssh:
            # 转换 https 为 git@github.com: 解决私有仓库鉴权问题
            repo_url = repo_url.replace("https://github.com/", "git@github.com:")
        return repo_url, match.group(2), match.group(3)
    return None, None, None
