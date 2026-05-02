"""Re-export of services.scoring.auth.

Importing from this module keeps the collections service self-contained at
the import-statement level without duplicating the GCP authentication code.
"""

from services.scoring.auth import (  # noqa: F401
    AuthenticationError,
    GCPAuthenticator,
    get_authenticated_storage_client,
)
