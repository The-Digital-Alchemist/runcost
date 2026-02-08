"""Utility functions and helpers."""

from pathlib import Path
from typing import Optional


def read_prompt_from_file(file_path: str) -> str:
    """Read prompt text from a file.

    Args:
        file_path: Path to file containing prompt

    Returns:
        Prompt text

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def format_cost(cost: float) -> str:
    """Format cost as currency string.

    Args:
        cost: Cost value

    Returns:
        Formatted cost string (e.g., "$0.00338")
    """
    return f"${cost:.5f}"


def format_tokens(tokens: int) -> str:
    """Format token count with thousands separator.

    Args:
        tokens: Token count

    Returns:
        Formatted token string (e.g., "1,250")
    """
    return f"{tokens:,}"


def format_percentage(percent: float, include_sign: bool = True) -> str:
    """Format percentage value.

    Args:
        percent: Percentage value
        include_sign: Include + sign for positive values

    Returns:
        Formatted percentage string (e.g., "+8.3%")
    """
    if include_sign and percent > 0:
        return f"+{percent:.1f}%"
    return f"{percent:.1f}%"


def get_confidence_emoji(confidence: str) -> str:
    """Get emoji for confidence level.

    Args:
        confidence: Confidence level ("High", "Medium", "Low")

    Returns:
        Emoji string
    """
    emoji_map = {
        "High": "🟢",
        "Medium": "🟡",
        "Low": "🔴",
    }
    return emoji_map.get(confidence, "⚪")


def format_box(
    content: dict,
    title: str,
    width: int = 40,
    use_unicode: bool = True
) -> str:
    """Format content in a box with unicode borders.

    Args:
        content: Dict of key-value pairs to display
        title: Box title
        width: Box width
        use_unicode: Use unicode box characters

    Returns:
        Formatted box string
    """
    if use_unicode:
        top_left, top_right = "╭", "╮"
        bottom_left, bottom_right = "╰", "╯"
        horizontal, vertical = "─", "│"
    else:
        top_left = top_right = bottom_left = bottom_right = "+"
        horizontal, vertical = "-", "|"

    lines = []

    # Top border with title
    title_text = f" {title} "
    padding = width - len(title_text) - 2
    left_pad = padding // 2
    right_pad = padding - left_pad
    lines.append(
        f"{top_left}{horizontal * left_pad}{title_text}{horizontal * right_pad}{top_right}"
    )

    # Content
    for key, value in content.items():
        line = f"{vertical} {key:<18} {value:>18} {vertical}"
        lines.append(line)

    # Bottom border
    lines.append(f"{bottom_left}{horizontal * (width - 2)}{bottom_right}")

    return "\n".join(lines)


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."
