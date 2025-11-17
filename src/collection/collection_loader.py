"""BGG Collection data loading for BGG predictive models."""

import logging
import os
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Union
from urllib.parse import urlencode

import polars as pl
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class BGGCollectionLoader:
    """Loader for BGG user collection data from XML API2."""

    def __init__(self, username: str, timeout: int = 30, max_retries: int = 3):
        """Initialize collection loader.

        Args:
            username: BGG username to load collection for
            timeout: Request timeout in seconds (default: 30)
            max_retries: Maximum number of retry attempts (default: 3)

        Raises:
            ValueError: If username is empty or BGG_API_TOKEN is not set
        """
        self.username = username.strip()
        self.base_url = "https://boardgamegeek.com/xmlapi2"
        self.timeout = timeout
        self.max_retries = max_retries

        # Load BGG API token from environment
        self.api_token = os.getenv("BGG_API_TOKEN")
        if not self.api_token:
            raise ValueError("BGG_API_TOKEN environment variable is required")

        if not self.username:
            raise ValueError("Username cannot be empty")

    def verify_collection_exists(self) -> bool:
        """Verify that the user's collection exists and is accessible.

        Returns:
            True if collection exists and is accessible, False otherwise
        """
        try:
            # Make a simple request to check if collection exists
            params = {
                "username": self.username,
                "brief": "1",  # Get brief results to minimize response size
            }

            response = self._make_request("/collection", params)

            if response is None:
                return False

            # Parse XML to check for errors
            try:
                root = ET.fromstring(response.content)

                # Check if we got an error response
                if root.tag == "errors":
                    logger.error(
                        f"BGG API error for user '{self.username}': {root.find('error').get('message', 'Unknown error')}"
                    )
                    return False

                # Check if we got a valid items response
                if root.tag == "items":
                    return True

                return False

            except ET.ParseError as e:
                logger.error(f"Error parsing XML response: {e}")
                return False

        except Exception as e:
            logger.error(f"Error verifying collection for user '{self.username}': {e}")
            return False

    def get_collection(self) -> Optional[pl.DataFrame]:
        """Retrieve the user's collection and return as a polars DataFrame.

        Returns:
            polars DataFrame with collection data, or None if failed
        """
        try:
            # Build parameters for collection request
            params = {
                "username": self.username,
            }

            logger.info(f"Fetching collection for user '{self.username}'")
            response = self._make_request("/collection", params)

            if response is None:
                return None

            # Parse XML response
            try:
                root = ET.fromstring(response.content)

                # Check for errors
                if root.tag == "errors":
                    error_msg = root.find("error").get("message", "Unknown error")
                    logger.error(f"BGG API error: {error_msg}")
                    return None

                # Parse collection items
                items = self._parse_collection_xml(root)

                if not items:
                    logger.warning(
                        f"No items found in collection for user '{self.username}'"
                    )
                    return None

                # Convert to polars DataFrame
                df = pl.DataFrame(items)
                logger.info(f"Successfully loaded {len(df)} items from collection")

                return df

            except ET.ParseError as e:
                logger.error(f"Error parsing XML response: {e}")
                return None

        except Exception as e:
            logger.error(f"Error retrieving collection for user '{self.username}': {e}")
            return None

    def _make_request(self, endpoint: str, params: Dict) -> Optional[requests.Response]:
        """Make HTTP request to BGG API with retry logic.

        Args:
            endpoint: API endpoint (e.g., '/collection')
            params: Query parameters

        Returns:
            Response object or None if failed
        """
        url = f"{self.base_url}{endpoint}"

        # Add authentication headers
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "User-Agent": "BGG-Predictive-Models/1.0",
        }

        for attempt in range(self.max_retries + 1):
            try:
                # Add delay between requests to be respectful to BGG API
                if attempt > 0:
                    delay = min(2**attempt, 10)  # Exponential backoff, max 10 seconds
                    logger.info(
                        f"Retrying request in {delay} seconds... (attempt {attempt + 1})"
                    )
                    time.sleep(delay)

                response = requests.get(
                    url, params=params, headers=headers, timeout=self.timeout
                )

                # Handle BGG-specific status codes
                if response.status_code == 202:
                    # BGG has queued the request, need to retry
                    logger.info(
                        "BGG has queued the request, waiting 5 seconds before retry..."
                    )
                    time.sleep(5)  # Wait longer for queued requests
                    continue
                elif response.status_code == 429:
                    logger.warning("Rate limited by BGG API, waiting before retry...")
                    time.sleep(5)
                    continue
                elif response.status_code >= 500:
                    logger.warning(f"Server error {response.status_code}, retrying...")
                    continue
                elif response.status_code == 200:
                    # Check if we got an error response in XML
                    if b"<error" in response.content:
                        logger.error(f"BGG API error in response: {response.text}")
                        return None
                    return response
                elif response.status_code == 401:
                    logger.error(
                        f"Unauthorized access (401) - check username '{self.username}'"
                    )
                    return None
                else:
                    logger.error(f"HTTP error {response.status_code}: {response.text}")
                    return None

            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1})")
                continue
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error (attempt {attempt + 1}): {e}")
                continue

        logger.error(f"Failed to make request after {self.max_retries + 1} attempts")
        return None

    def _parse_collection_xml(self, root: ET.Element) -> List[Dict]:
        """Parse XML collection response into list of dictionaries.

        Args:
            root: XML root element

        Returns:
            List of dictionaries containing game data
        """
        items = []

        for item in root.findall("item"):
            try:
                # Basic item information
                game_data = {
                    "game_id": int(item.get("objectid")),
                    "game_name": (
                        item.find("name").text
                        if item.find("name") is not None
                        else None
                    ),
                    "subtype": item.get("subtype"),
                    "collection_id": (
                        int(item.get("collid")) if item.get("collid") else None
                    ),
                }

                # Status information (owned, previously owned, etc.)
                status = item.find("status")
                if status is not None:
                    game_data.update(
                        {
                            "owned": status.get("own") == "1",
                            "previously_owned": status.get("prevowned") == "1",
                            "for_trade": status.get("fortrade") == "1",
                            "want": status.get("want") == "1",
                            "want_to_play": status.get("wanttoplay") == "1",
                            "want_to_buy": status.get("wanttobuy") == "1",
                            "wishlist": status.get("wishlist") == "1",
                            "wishlist_priority": (
                                int(status.get("wishlistpriority"))
                                if status.get("wishlistpriority")
                                else None
                            ),
                            "preordered": status.get("preordered") == "1",
                            "last_modified": status.get("lastmodified"),
                        }
                    )

                # User rating and comments
                game_data.update(
                    {
                        "user_rating": (
                            float(item.find("rating").get("value"))
                            if item.find("rating") is not None
                            and item.find("rating").get("value") != "N/A"
                            else None
                        ),
                        "user_comment": (
                            item.find("comment").text
                            if item.find("comment") is not None
                            else None
                        ),
                    }
                )

                items.append(game_data)

            except (ValueError, AttributeError) as e:
                logger.warning(
                    f"Error parsing item {item.get('objectid', 'unknown')}: {e}"
                )
                continue

        return items

    def get_collection_summary(self) -> Optional[Dict]:
        """Get summary statistics about the user's collection.

        Returns:
            Dictionary with collection summary or None if failed
        """
        df = self.get_collection()

        if df is None:
            return None

        try:
            summary = {
                "username": self.username,
                "total_items": len(df),
                "owned_games": (
                    len(df.filter(pl.col("owned") == True))
                    if "owned" in df.columns
                    else 0
                ),
                "unique_games": (
                    len(df.filter(pl.col("subtype") == "boardgame"))
                    if "subtype" in df.columns
                    else 0
                ),
                "expansions": (
                    len(df.filter(pl.col("subtype") == "boardgameexpansion"))
                    if "subtype" in df.columns
                    else 0
                ),
                "avg_user_rating": (
                    df.select(pl.col("user_rating").mean()).item()
                    if "user_rating" in df.columns
                    else None
                ),
                "top_rated_game": None,
            }

            # Get top rated game by user
            if "user_rating" in df.columns and "game_name" in df.columns:
                top_rated = (
                    df.filter(pl.col("user_rating").is_not_null())
                    .sort("user_rating", descending=True)
                    .head(1)
                )
                if len(top_rated) > 0:
                    summary["top_rated_game"] = {
                        "name": top_rated.select("game_name").item(),
                        "rating": top_rated.select("user_rating").item(),
                    }

            return summary

        except Exception as e:
            logger.error(f"Error generating collection summary: {e}")
            return None
