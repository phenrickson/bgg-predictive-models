"""End-to-end smoke tests against real BGG users and real BigQuery.

Not hermetic: these leave real user rows in `collections.user_collections`.
That's intentional — the upsert model means repeated runs are idempotent,
and the rows represent actual BGG state we want stored.

Gated behind the `integration` marker. Run with:

    pytest -m integration tests/test_collection_smoke.py -v
"""

import pytest

from src.collection.collection_loader import BGGCollectionLoader
from src.collection.collection_storage import CollectionStorage


pytestmark = pytest.mark.integration


SMOKE_USERS = ["phenrickson", "GOBBluth89"]


@pytest.fixture(scope="module")
def storage():
    return CollectionStorage(environment="dev")


@pytest.mark.parametrize("username", SMOKE_USERS)
def test_real_user_roundtrip(username: str, storage: CollectionStorage):
    """Pull -> save -> read -> re-save is idempotent for a real user."""
    df = BGGCollectionLoader(username).get_collection()
    assert df is not None, f"loader returned None for {username}"
    assert df.height > 0, f"loader returned 0 rows for {username}"

    # Request-side filter (Task 4): no expansions should ever appear.
    subtypes = set(df["subtype"].unique().to_list())
    assert subtypes == {"boardgame"}, (
        f"unexpected subtypes in {username}'s pull: {subtypes}"
    )

    storage.save_collection(username, df)
    first = storage.get_latest_collection(username)
    assert first is not None
    assert first.height == df.height

    storage.save_collection(username, df)
    second = storage.get_latest_collection(username)
    assert second is not None
    assert second.height == first.height, (
        f"row count changed on idempotent re-pull for {username}: "
        f"{first.height} -> {second.height}"
    )

    # first_seen_at must not move on re-pull; updated_at must advance.
    first_by_id = {r["game_id"]: r for r in first.iter_rows(named=True)}
    second_by_id = {r["game_id"]: r for r in second.iter_rows(named=True)}
    for gid in first_by_id:
        assert first_by_id[gid]["first_seen_at"] == second_by_id[gid]["first_seen_at"]
        assert second_by_id[gid]["updated_at"] >= first_by_id[gid]["updated_at"]


def test_cross_user_isolation_with_real_users(storage: CollectionStorage):
    """After both users are saved, each still sees only their own rows."""
    counts = {}
    for username in SMOKE_USERS:
        df = BGGCollectionLoader(username).get_collection()
        assert df is not None, f"loader returned None for {username}"
        storage.save_collection(username, df)
        counts[username] = df.height

    for username, expected in counts.items():
        active = storage.get_latest_collection(username)
        assert active is not None
        assert active.height == expected, (
            f"{username}: expected {expected} active rows, got {active.height}"
        )
        # Every row belongs to this user.
        users = set(active["username"].unique().to_list())
        assert users == {username}, f"cross-user contamination: {users}"
