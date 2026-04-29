"""CLI: batch-register one candidate across multiple outcomes for a user.

Continue-on-error: if one outcome fails to register, the rest still run; exits
non-zero at the end if any failed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from services.collections.register_model import register_collection

logger = logging.getLogger(__name__)


def register_many(
    *,
    username: str,
    outcomes: List[str],
    candidate: str,
    description: str,
    version: str = "latest",
    environment: str = "dev",
    local_root: str = "models/collections",
) -> dict:
    """Register one candidate across multiple outcomes. Returns
    ``{"succeeded": [...], "failed": [(outcome, error_msg), ...]}``.
    """
    succeeded: List[dict] = []
    failed: List[tuple] = []

    for outcome in outcomes:
        try:
            registration = register_collection(
                username=username,
                outcome=outcome,
                candidate=candidate,
                description=description,
                version=version,
                environment=environment,
                local_root=local_root,
            )
            succeeded.append(registration)
            logger.info(
                "Registered %s/%s candidate=%s as v%d",
                username, outcome, candidate, registration["version"],
            )
        except (FileNotFoundError, ValueError) as e:
            failed.append((outcome, str(e)))
            logger.error("Failed to register %s/%s: %s", username, outcome, e)

    return {"succeeded": succeeded, "failed": failed}


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--username", required=True)
    p.add_argument(
        "--outcomes",
        required=True,
        help="Comma-separated outcomes (e.g. 'own,ever_owned,rated')",
    )
    p.add_argument("--candidate", required=True)
    p.add_argument("--description", required=True)
    p.add_argument("--version", default="latest")
    p.add_argument("--environment", default="dev")
    p.add_argument("--local-root", default="models/collections")
    args = p.parse_args(argv)

    outcomes = [o.strip() for o in args.outcomes.split(",") if o.strip()]
    if not outcomes:
        print("error: --outcomes must be non-empty", file=sys.stderr)
        return 1

    result = register_many(
        username=args.username,
        outcomes=outcomes,
        candidate=args.candidate,
        description=args.description,
        version=args.version,
        environment=args.environment,
        local_root=args.local_root,
    )

    print(f"\nRegistered: {len(result['succeeded'])} / {len(outcomes)}")
    for r in result["succeeded"]:
        print(f"  ✓ {r['outcome']} v{r['version']}")
    for outcome, err in result["failed"]:
        print(f"  ✗ {outcome}: {err}")

    return 1 if result["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
