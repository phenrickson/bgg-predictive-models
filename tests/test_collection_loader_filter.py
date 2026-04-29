"""Unit test that BGGCollectionLoader requests exclude expansions."""

from unittest.mock import patch, MagicMock

from src.collection.collection_loader import BGGCollectionLoader


def test_get_collection_excludes_expansions_in_request(monkeypatch):
    """Verify the request params include excludesubtype=boardgameexpansion."""
    monkeypatch.setenv("BGG_API_TOKEN", "fake-token-for-test")
    loader = BGGCollectionLoader(username="anyuser")

    captured_params = {}

    def fake_make_request(endpoint, params):
        captured_params.update(params)
        mock_resp = MagicMock()
        mock_resp.content = b"<items totalitems='0'></items>"
        return mock_resp

    with patch.object(loader, "_make_request", side_effect=fake_make_request):
        loader.get_collection()

    assert captured_params.get("subtype") == "boardgame"
    assert captured_params.get("excludesubtype") == "boardgameexpansion"
