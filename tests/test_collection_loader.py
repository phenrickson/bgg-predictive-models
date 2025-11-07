"""Tests for BGGCollectionLoader."""

import pytest
import polars as pl
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET
import os

from src.data.collection_loader import BGGCollectionLoader


class TestBGGCollectionLoader:
    """Test cases for BGGCollectionLoader."""

    @patch.dict(os.environ, {"BGG_API_TOKEN": "test_token_123"})
    def test_init_valid_username(self):
        """Test initialization with valid username."""
        loader = BGGCollectionLoader("phenrickson")
        assert loader.username == "phenrickson"
        assert loader.base_url == "https://boardgamegeek.com/xmlapi2"
        assert loader.timeout == 30
        assert loader.max_retries == 3
        assert loader.api_token == "test_token_123"

    def test_init_missing_token(self):
        """Test initialization without BGG_API_TOKEN raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="BGG_API_TOKEN environment variable is required"
            ):
                BGGCollectionLoader("phenrickson")

    @patch.dict(os.environ, {"BGG_API_TOKEN": "test_token_123"})
    def test_init_empty_username(self):
        """Test initialization with empty username raises ValueError."""
        with pytest.raises(ValueError, match="Username cannot be empty"):
            BGGCollectionLoader("")

    @patch.dict(os.environ, {"BGG_API_TOKEN": "test_token_123"})
    def test_init_whitespace_username(self):
        """Test initialization strips whitespace from username."""
        loader = BGGCollectionLoader("  phenrickson  ")
        assert loader.username == "phenrickson"

    @patch.dict(os.environ, {"BGG_API_TOKEN": "test_token_123"})
    def test_init_custom_parameters(self):
        """Test initialization with custom timeout and retries."""
        loader = BGGCollectionLoader("phenrickson", timeout=60, max_retries=5)
        assert loader.timeout == 60
        assert loader.max_retries == 5

    @patch("src.data.collection_loader.requests.get")
    @patch.dict(os.environ, {"BGG_API_TOKEN": "test_token_123"})
    def test_verify_collection_exists_success(self, mock_get):
        """Test successful collection verification."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'<items totalitems="1"><item objectid="1" /></items>'
        mock_get.return_value = mock_response

        loader = BGGCollectionLoader("phenrickson")
        result = loader.verify_collection_exists()

        assert result is True
        mock_get.assert_called_once()

        # Verify that the request was made with authentication headers
        call_args = mock_get.call_args
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer test_token_123"
        assert "User-Agent" in headers

    @patch("src.data.collection_loader.requests.get")
    @patch.dict(os.environ, {"BGG_API_TOKEN": "test_token_123"})
    def test_verify_collection_exists_error_response(self, mock_get):
        """Test collection verification with error response."""
        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'<errors><error message="Invalid username" /></errors>'
        mock_get.return_value = mock_response

        loader = BGGCollectionLoader("invaliduser")
        result = loader.verify_collection_exists()

        assert result is False

    @patch("src.data.collection_loader.requests.get")
    @patch.dict(os.environ, {"BGG_API_TOKEN": "test_token_123"})
    def test_verify_collection_exists_network_error(self, mock_get):
        """Test collection verification with network error."""
        mock_get.side_effect = Exception("Network error")

        loader = BGGCollectionLoader("phenrickson")
        result = loader.verify_collection_exists()

        assert result is False

    @patch("src.data.collection_loader.requests.get")
    @patch.dict(os.environ, {"BGG_API_TOKEN": "test_token_123"})
    def test_get_collection_success(self, mock_get):
        """Test successful collection retrieval."""
        # Mock successful response with sample data
        sample_xml = """
        <items totalitems="2">
            <item objectid="174430" subtype="boardgame" collid="12345">
                <name>Gloomhaven</name>
                <status own="1" prevowned="0" fortrade="0" want="0" wanttoplay="0" wanttobuy="0" wishlist="0" preordered="0" lastmodified="2023-01-01 12:00:00" />
                <rating value="9.5" />
                <comment>Amazing game!</comment>
                <stats minplayers="1" maxplayers="4" minplaytime="60" maxplaytime="120" playingtime="90" numowned="50000">
                    <rating usersrated="45000" average="8.8" bayesaverage="8.7" stddev="1.2" median="9.0">
                        <ranks>
                            <rank type="subtype" name="boardgame" value="1" />
                        </ranks>
                    </rating>
                </stats>
            </item>
            <item objectid="161936" subtype="boardgame" collid="12346">
                <name>Pandemic Legacy: Season 1</name>
                <status own="1" prevowned="0" fortrade="0" want="0" wanttoplay="0" wanttobuy="0" wishlist="0" preordered="0" lastmodified="2023-01-02 12:00:00" />
                <rating value="8.0" />
                <stats minplayers="2" maxplayers="4" minplaytime="60" maxplaytime="60" playingtime="60" numowned="30000">
                    <rating usersrated="25000" average="8.6" bayesaverage="8.5" stddev="1.1" median="8.5">
                        <ranks>
                            <rank type="subtype" name="boardgame" value="5" />
                        </ranks>
                    </rating>
                </stats>
            </item>
        </items>
        """

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = sample_xml.encode("utf-8")
        mock_get.return_value = mock_response

        loader = BGGCollectionLoader("phenrickson")
        df = loader.get_collection()

        assert df is not None
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2

        # Check that expected columns exist
        expected_columns = ["game_id", "game_name", "subtype", "owned", "user_rating"]
        for col in expected_columns:
            assert col in df.columns

        # Check specific values
        assert (
            df.filter(pl.col("game_id") == 174430).select("game_name").item()
            == "Gloomhaven"
        )
        assert (
            df.filter(pl.col("game_id") == 174430).select("user_rating").item() == 9.5
        )

    @patch("src.data.collection_loader.requests.get")
    @patch.dict(os.environ, {"BGG_API_TOKEN": "test_token_123"})
    def test_get_collection_empty_response(self, mock_get):
        """Test collection retrieval with empty collection."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'<items totalitems="0"></items>'
        mock_get.return_value = mock_response

        loader = BGGCollectionLoader("phenrickson")
        df = loader.get_collection()

        assert df is None

    @patch("src.data.collection_loader.requests.get")
    @patch.dict(os.environ, {"BGG_API_TOKEN": "test_token_123"})
    def test_get_collection_with_filters(self, mock_get):
        """Test collection retrieval with filters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'<items totalitems="1"><item objectid="1" /></items>'
        mock_get.return_value = mock_response

        loader = BGGCollectionLoader("phenrickson")
        loader.get_collection(include_expansions=False, own_only=True)

        # Verify the request was made with correct parameters
        call_args = mock_get.call_args
        params = call_args[1]["params"]

        assert params["subtype"] == "boardgame"  # No expansions
        assert params["own"] == "1"  # Only owned games

    @patch.dict(os.environ, {"BGG_API_TOKEN": "test_token_123"})
    def test_parse_collection_xml_basic(self):
        """Test XML parsing with basic item data."""
        xml_content = """
        <items totalitems="1">
            <item objectid="174430" subtype="boardgame" collid="12345">
                <name>Gloomhaven</name>
                <status own="1" />
                <rating value="9.5" />
            </item>
        </items>
        """

        root = ET.fromstring(xml_content)
        loader = BGGCollectionLoader("phenrickson")
        items = loader._parse_collection_xml(root)

        assert len(items) == 1
        item = items[0]
        assert item["game_id"] == 174430
        assert item["game_name"] == "Gloomhaven"
        assert item["subtype"] == "boardgame"
        assert item["owned"] is True
        assert item["user_rating"] == 9.5

    def test_parse_collection_xml_missing_elements(self):
        """Test XML parsing with missing optional elements."""
        xml_content = """
        <items totalitems="1">
            <item objectid="174430" subtype="boardgame">
                <name>Gloomhaven</name>
            </item>
        </items>
        """

        root = ET.fromstring(xml_content)
        loader = BGGCollectionLoader("phenrickson")
        items = loader._parse_collection_xml(root)

        assert len(items) == 1
        item = items[0]
        assert item["game_id"] == 174430
        assert item["collection_id"] is None
        assert item["user_rating"] is None

    @patch.object(BGGCollectionLoader, "get_collection")
    def test_get_collection_summary_success(self, mock_get_collection):
        """Test successful collection summary generation."""
        # Mock collection data
        mock_df = pl.DataFrame(
            {
                "game_id": [1, 2, 3],
                "game_name": ["Game A", "Game B", "Game C"],
                "subtype": ["boardgame", "boardgame", "boardgameexpansion"],
                "owned": [True, True, False],
                "user_rating": [8.0, 9.0, None],
                "average_rating": [7.5, 8.5, 7.0],
            }
        )
        mock_get_collection.return_value = mock_df

        loader = BGGCollectionLoader("phenrickson")
        summary = loader.get_collection_summary()

        assert summary is not None
        assert summary["username"] == "phenrickson"
        assert summary["total_items"] == 3
        assert summary["owned_games"] == 2
        assert summary["unique_games"] == 2
        assert summary["expansions"] == 1
        assert summary["top_rated_game"]["name"] == "Game B"
        assert summary["top_rated_game"]["rating"] == 9.0

    @patch.object(BGGCollectionLoader, "get_collection")
    def test_get_collection_summary_no_collection(self, mock_get_collection):
        """Test collection summary when collection retrieval fails."""
        mock_get_collection.return_value = None

        loader = BGGCollectionLoader("phenrickson")
        summary = loader.get_collection_summary()

        assert summary is None


class TestBGGCollectionLoaderIntegration:
    """Integration tests for BGGCollectionLoader with real API calls."""

    @pytest.mark.integration
    def test_verify_collection_exists_real_user(self):
        """Test collection verification with real BGG user 'phenrickson'."""
        loader = BGGCollectionLoader("phenrickson")
        result = loader.verify_collection_exists()

        # This should pass if 'phenrickson' is a valid BGG user with a collection
        # If it fails with 401, the user might not exist or have privacy settings
        if not result:
            pytest.skip(
                "User 'phenrickson' collection not accessible (may not exist or be private)"
            )

        assert result is True

    @pytest.mark.integration
    def test_verify_collection_exists_invalid_user(self):
        """Test collection verification with invalid user."""
        loader = BGGCollectionLoader("thisuserdoesnotexist12345")
        result = loader.verify_collection_exists()

        assert result is False

    @pytest.mark.integration
    def test_get_collection_real_user(self):
        """Test collection retrieval with real BGG user 'phenrickson'."""
        loader = BGGCollectionLoader("phenrickson")
        df = loader.get_collection(own_only=True)

        if df is not None:
            assert isinstance(df, pl.DataFrame)
            assert len(df) > 0

            # Check that basic columns exist
            expected_columns = ["game_id", "game_name", "owned"]
            for col in expected_columns:
                assert col in df.columns

            # All games should be owned since we used own_only=True
            if "owned" in df.columns:
                assert df.select(pl.col("owned").all()).item()

    @pytest.mark.integration
    def test_get_collection_summary_real_user(self):
        """Test collection summary with real BGG user 'phenrickson'."""
        loader = BGGCollectionLoader("phenrickson")
        summary = loader.get_collection_summary()

        if summary is not None:
            assert summary["username"] == "phenrickson"
            assert isinstance(summary["total_items"], int)
            assert summary["total_items"] >= 0
