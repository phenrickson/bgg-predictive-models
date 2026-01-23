"""Text tokenization utilities."""

import html
import re
from typing import List

# Domain-specific text normalizations (applied before tokenization)
TEXT_NORMALIZATIONS = {
    # Spelling variants
    "co-operative": "cooperative",
    "co-operatively": "cooperatively",
    "co-operation": "cooperation",
    "co-op": "coop",
    "sci-fi": "scifi",
    "semi-cooperative": "semicooperative",
    "multi-player": "multiplayer",
    "real-time": "realtime",
    "turn-based": "turnbased",
    "role-playing": "roleplaying",
    "role-play": "roleplay",
    # Game mechanics
    "deck-building": "deckbuilding",
    "deck-builder": "deckbuilder",
    # Game terms
    "re-play": "replay",
    "pre-order": "preorder",
    "mid-game": "midgame",
    "end-game": "endgame",
    "start-player": "startplayer",
    # Player counts (word form)
    "two-player": "twoplayer",
    "three-player": "threeplayer",
    "four-player": "fourplayer",
    "five-player": "fiveplayer",
    "six-player": "sixplayer",
    # Player counts (numeric form - convert to words so they survive tokenization)
    "1-player": "oneplayer",
    "2-player": "twoplayer",
    "3-player": "threeplayer",
    "4-player": "fourplayer",
    "5-player": "fiveplayer",
    "6-player": "sixplayer",
    "7-player": "sevenplayer",
    "8-player": "eightplayer",
}


NUM_TO_WORD = {
    "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
}


def _normalize_player_ranges(text: str) -> str:
    """Convert player count ranges like '2-4 players' to 'twotofourplayers'."""
    def replace_range(match):
        start, end = match.group(1), match.group(2)
        start_word = NUM_TO_WORD.get(start, start)
        end_word = NUM_TO_WORD.get(end, end)
        return f"{start_word}to{end_word}players"

    # Match patterns like "2-4 players", "1-8 players", etc.
    return re.sub(r"(\d+)-(\d+)\s*players?", replace_range, text)


def _normalize_text(text: str) -> str:
    """Apply domain-specific text normalizations."""
    text_lower = text.lower()

    # Apply player range normalization first (regex-based)
    text_lower = _normalize_player_ranges(text_lower)

    # Then apply simple string replacements
    for pattern, replacement in TEXT_NORMALIZATIONS.items():
        text_lower = text_lower.replace(pattern, replacement)

    return text_lower


def tokenize(text: str, min_length: int = 2) -> List[str]:
    """Tokenize text into lowercase words.

    Args:
        text: Input text.
        min_length: Minimum word length to keep.

    Returns:
        List of tokens.
    """
    if not text:
        return []

    # Decode HTML entities (e.g., &aacute; -> á, &amp; -> &)
    text = html.unescape(text)

    # Apply domain-specific normalizations (also lowercases)
    text = _normalize_text(text)
    words = re.findall(r"[a-záàâäãåæçéèêëíìîïñóòôöõøúùûüýÿ]+", text)

    # Filter by length
    return [w for w in words if len(w) >= min_length]


def tokenize_documents(
    documents: List[str], min_length: int = 2
) -> List[List[str]]:
    """Tokenize a list of documents.

    Args:
        documents: List of text documents.
        min_length: Minimum word length to keep.

    Returns:
        List of token lists.
    """
    return [tokenize(doc, min_length) for doc in documents]
