from agent_sync.fetcher import parse_github_url


def test_parse_github_url():
    url = "https://github.com/org/repo/tree/main/skills"
    repo_url, branch, path = parse_github_url(url)
    assert repo_url == "https://github.com/org/repo.git"
    assert branch == "main"
    assert path == "skills"
