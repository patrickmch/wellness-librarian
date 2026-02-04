"""
Self-correction critic layer for wellness responses.

Verifies that health claims in responses are supported by source excerpts.
Corrects or flags unsupported claims to prevent hallucinations.
"""

import logging
from typing import Optional

from backend.config import get_settings
from backend.rag.generator import get_anthropic_client, get_async_anthropic_client

logger = logging.getLogger(__name__)

CRITIC_SYSTEM_PROMPT = """You are a fact-checker for wellness content. Verify that health claims are supported by the sources.

CRITICAL OUTPUT FORMAT:
- If accurate: Output ONLY the word "VERIFIED"
- If corrections needed: Output ONLY the corrected response text

FORBIDDEN - NEVER include any of these in your output:
- Explanations of what you changed
- Meta-commentary like "The revised response..." or "I have removed..."
- Analysis of accuracy
- Notes or disclaimers about your edits
- ANYTHING other than VERIFIED or the pure corrected response"""

CRITIC_USER_PROMPT = """Verify this wellness response against the sources.

RESPONSE:
{response}

SOURCES:
{sources}

RULES:
- Specific dosages/supplements/timing MUST be in sources to be valid
- General wellness advice can be paraphrased
- If mostly accurate with minor issues, output VERIFIED
- Only correct if there are clear unsupported health claims

YOUR OUTPUT MUST BE EXACTLY ONE OF:
1. The single word VERIFIED (nothing else, no explanation)
2. The corrected response text ONLY (no explanation of changes, no meta-commentary)

DO NOT explain what you changed. DO NOT add notes. Just output the corrected text."""


def _strip_trailing_analysis(text: str) -> str:
    """
    Strip trailing meta-analysis that the critic sometimes adds.

    Detects patterns like:
    - "The revised response..."
    - "I have removed..."
    - "Note: ..."
    """
    import re

    # Patterns that indicate meta-commentary (case insensitive)
    analysis_patterns = [
        r'\n\n(?:The revised response|This response|I have|Note:|I\'ve|The above|This revised)',
        r'\n\n(?:---+)\s*\n.*$',  # Horizontal rule followed by analysis
    ]

    for pattern in analysis_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            text = text[:match.start()].rstrip()

    return text


async def verify_response_async(
    response: str,
    sources: list[dict],
    max_excerpt_chars: int = 600,
) -> tuple[str, bool]:
    """
    Verify a response against source excerpts using a critic model.

    Args:
        response: The generated response to verify
        sources: List of source dicts with 'excerpt' and 'title' fields
        max_excerpt_chars: Max chars per excerpt to include

    Returns:
        Tuple of (final_response, was_corrected)
        - If verified: (original_response, False)
        - If corrected: (corrected_response, True)
    """
    settings = get_settings()

    # Build source excerpts for verification
    excerpt_parts = []
    for i, source in enumerate(sources, 1):
        excerpt = source.get("excerpt", "")[:max_excerpt_chars]
        title = source.get("title", "Unknown")
        excerpt_parts.append(f"[{i}] {title}:\n{excerpt}")

    sources_text = "\n\n".join(excerpt_parts)

    # Build the verification prompt
    user_prompt = CRITIC_USER_PROMPT.format(
        response=response,
        sources=sources_text,
    )

    try:
        client = get_async_anthropic_client()

        # Use haiku for speed and cost efficiency
        message = await client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=2000,
            system=CRITIC_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        critic_output = message.content[0].text.strip()

        # Check if verified (exact match or starts with VERIFIED)
        if critic_output.upper() == "VERIFIED" or critic_output.upper().startswith("VERIFIED"):
            logger.info("[critic] Response verified as accurate")
            return response, False
        else:
            # Strip any trailing meta-analysis the critic might have added
            corrected = _strip_trailing_analysis(critic_output)
            logger.info("[critic] Response corrected by critic")
            return corrected, True

    except Exception as e:
        logger.warning(f"[critic] Verification failed, using original: {e}")
        return response, False


def verify_response(
    response: str,
    sources: list[dict],
    max_excerpt_chars: int = 600,
) -> tuple[str, bool]:
    """
    Synchronous version of verify_response_async.

    Args:
        response: The generated response to verify
        sources: List of source dicts with 'excerpt' and 'title' fields
        max_excerpt_chars: Max chars per excerpt to include

    Returns:
        Tuple of (final_response, was_corrected)
    """
    settings = get_settings()

    # Build source excerpts for verification
    excerpt_parts = []
    for i, source in enumerate(sources, 1):
        excerpt = source.get("excerpt", "")[:max_excerpt_chars]
        title = source.get("title", "Unknown")
        excerpt_parts.append(f"[{i}] {title}:\n{excerpt}")

    sources_text = "\n\n".join(excerpt_parts)

    user_prompt = CRITIC_USER_PROMPT.format(
        response=response,
        sources=sources_text,
    )

    try:
        client = get_anthropic_client()

        message = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=2000,
            system=CRITIC_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        critic_output = message.content[0].text.strip()

        # Check if verified (exact match or starts with VERIFIED)
        if critic_output.upper() == "VERIFIED" or critic_output.upper().startswith("VERIFIED"):
            logger.info("[critic] Response verified as accurate")
            return response, False
        else:
            # Strip any trailing meta-analysis the critic might have added
            corrected = _strip_trailing_analysis(critic_output)
            logger.info("[critic] Response corrected by critic")
            return corrected, True

    except Exception as e:
        logger.warning(f"[critic] Verification failed, using original: {e}")
        return response, False
