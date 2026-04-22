import re


def parse_github_url(url: str):
    match = re.match(r"(https://github\.com/[^/]+/[^/]+)/tree/([^/]+)/(.*)", url)
    if match:
        return match.group(1) + ".git", match.group(2), match.group(3)
    return None, None, None
