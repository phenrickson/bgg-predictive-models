"""Authentication utilities for the game embeddings service.

Re-exports from services.scoring.auth for consistency.
"""

from services.scoring.auth import (  # noqa: F401
    AuthenticationError,
    GCPAuthenticator,
    get_authenticated_storage_client,
    verify_authentication,
)

__all__ = [
    "AuthenticationError",
    "GCPAuthenticator",
    "get_authenticated_storage_client",
    "verify_authentication",
]
