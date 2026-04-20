"""Authentication utilities for the embeddings service.

Re-exports from scoring_service.auth for consistency.
"""

import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from scoring_service.auth import (  # noqa: E402, F401
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
