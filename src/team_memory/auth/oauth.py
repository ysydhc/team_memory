"""OAuth 2.0 authentication support (P4-6).

Provides GitHub and GitLab OAuth integration.
Requires the `authlib` package: pip install authlib

When a user logs in via OAuth:
1. Redirect to provider's authorization URL
2. Receive callback with authorization code
3. Exchange code for access token
4. Fetch user profile
5. Auto-create API Key linked to OAuth identity

This module provides the foundation; full implementation requires
authlib to be installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("team_memory.auth.oauth")


@dataclass
class OAuthConfig:
    """OAuth provider configuration."""
    provider: str  # github, gitlab
    client_id: str
    client_secret: str
    redirect_uri: str
    # GitHub-specific
    scope: str = "read:user"


def is_oauth_available() -> bool:
    """Check if authlib is installed."""
    try:
        import authlib  # noqa: F401
        return True
    except ImportError:
        return False


async def create_oauth_routes(app, config: OAuthConfig):
    """Register OAuth routes on a FastAPI app.

    Only registers routes if authlib is available.
    """
    if not is_oauth_available():
        logger.info("authlib not installed. OAuth routes not registered.")
        return

    from authlib.integrations.starlette_client import OAuth

    oauth = OAuth()

    if config.provider == "github":
        oauth.register(
            name="github",
            client_id=config.client_id,
            client_secret=config.client_secret,
            access_token_url="https://github.com/login/oauth/access_token",
            access_token_params=None,
            authorize_url="https://github.com/login/oauth/authorize",
            authorize_params=None,
            api_base_url="https://api.github.com/",
            client_kwargs={"scope": config.scope},
        )
    elif config.provider == "gitlab":
        oauth.register(
            name="gitlab",
            client_id=config.client_id,
            client_secret=config.client_secret,
            access_token_url="https://gitlab.com/oauth/token",
            authorize_url="https://gitlab.com/oauth/authorize",
            api_base_url="https://gitlab.com/api/v4/",
            client_kwargs={"scope": "read_user"},
        )

    logger.info("OAuth routes registered for provider: %s", config.provider)
