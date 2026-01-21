"""
RAG response generation using Claude.
Assembles context from retrieved chunks and generates helpful responses.
"""

import logging
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from anthropic import Anthropic, AsyncAnthropic

from backend.config import get_settings
from backend.rag.retriever import retrieve, RetrievalResponse

logger = logging.getLogger(__name__)


# The Wellness Librarian persona prompt
SYSTEM_PROMPT = """You are the Wellness Librarian, a warm and knowledgeable guide to the Wellness Evolution Community's video content library. You help community members discover relevant videos and understand wellness concepts discussed in the transcripts.

## Your Role
- You are a helpful librarian who knows the content of the video library deeply
- You guide people to the most relevant videos for their questions
- You synthesize information from multiple videos when appropriate
- You speak with warmth, encouragement, and genuine care for the person's wellness journey

## Guidelines

### When Answering Questions
1. **Ground your responses in the provided transcript excerpts** - Only share information that appears in the context provided
2. **Cite your sources** - Mention which video(s) the information comes from (use the video titles)
3. **Be honest about limitations** - If the provided context doesn't contain information about a topic, say so kindly and suggest they explore other resources
4. **Synthesize, don't just quote** - Weave the information into a coherent, helpful response
5. **Encourage exploration** - When relevant, suggest they watch the full video for more depth

### Tone & Style
- Warm and supportive, like a knowledgeable friend
- Use accessible language, avoiding unnecessary jargon
- Be encouraging about the person's wellness journey
- Keep responses focused and helpful, not overly long

### When You Don't Know
If the context doesn't contain relevant information, respond honestly:
"I don't see information about that specific topic in the videos I have access to. However, you might find related content in [suggest a category if relevant], or this might be something to explore with a practitioner directly."

### Format
- Use natural paragraphs for explanations
- Include video titles when referencing specific content
- Keep responses concise but complete (aim for 2-4 paragraphs typically)
- End with an encouraging note or invitation to explore further when appropriate"""


def build_context_prompt(retrieval: RetrievalResponse, user_message: str) -> str:
    """
    Build the user prompt with retrieved context.

    Args:
        retrieval: RetrievalResponse with search results
        user_message: The user's question

    Returns:
        Formatted prompt string
    """
    if not retrieval.results:
        context_section = "No relevant video content was found for this query."
    else:
        context_parts = []
        for i, result in enumerate(retrieval.results, 1):
            context_parts.append(
                f"[Video {i}: \"{result.title}\" ({result.category}) - {result.metadata.get('duration', '')}]\n"
                f"{result.text}"
            )
        context_section = "\n\n---\n\n".join(context_parts)

    return f"""## Retrieved Video Transcripts

{context_section}

---

## User's Question

{user_message}

Please provide a helpful response based on the video content above. Remember to cite which videos your information comes from."""


@dataclass
class GenerationResult:
    """Result from RAG generation."""
    response: str
    sources: list[dict]
    retrieval_count: int
    model_used: str


def get_anthropic_client() -> Anthropic:
    """Get synchronous Anthropic client."""
    settings = get_settings()
    return Anthropic(api_key=settings.anthropic_api_key)


def get_async_anthropic_client() -> AsyncAnthropic:
    """Get asynchronous Anthropic client."""
    settings = get_settings()
    return AsyncAnthropic(api_key=settings.anthropic_api_key)


def generate_response(
    user_message: str,
    category: Optional[str] = None,
    top_k: int | None = None,
) -> GenerationResult:
    """
    Generate a response to the user's question using RAG.

    Args:
        user_message: The user's question
        category: Optional category filter
        top_k: Number of chunks to retrieve

    Returns:
        GenerationResult with response and sources
    """
    settings = get_settings()
    top_k = top_k or settings.default_top_k

    # Retrieve relevant context
    retrieval = retrieve(
        query=user_message,
        top_k=top_k,
        category=category,
    )

    logger.info(f"Retrieved {retrieval.total_results} chunks for query: {user_message[:50]}...")

    # Build prompt with context
    user_prompt = build_context_prompt(retrieval, user_message)

    # Generate response with Claude
    client = get_anthropic_client()

    message = client.messages.create(
        model=settings.claude_model,
        max_tokens=settings.max_response_tokens,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_prompt}
        ],
    )

    response_text = message.content[0].text

    return GenerationResult(
        response=response_text,
        sources=retrieval.get_sources(),
        retrieval_count=retrieval.total_results,
        model_used=settings.claude_model,
    )


async def generate_response_async(
    user_message: str,
    category: Optional[str] = None,
    top_k: int | None = None,
) -> GenerationResult:
    """
    Generate a response asynchronously.

    Args:
        user_message: The user's question
        category: Optional category filter
        top_k: Number of chunks to retrieve

    Returns:
        GenerationResult with response and sources
    """
    settings = get_settings()
    top_k = top_k or settings.default_top_k

    # Retrieve relevant context (sync for now, could be async)
    retrieval = retrieve(
        query=user_message,
        top_k=top_k,
        category=category,
    )

    logger.info(f"Retrieved {retrieval.total_results} chunks for query: {user_message[:50]}...")

    # Build prompt with context
    user_prompt = build_context_prompt(retrieval, user_message)

    # Generate response with Claude
    client = get_async_anthropic_client()

    message = await client.messages.create(
        model=settings.claude_model,
        max_tokens=settings.max_response_tokens,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_prompt}
        ],
    )

    response_text = message.content[0].text

    return GenerationResult(
        response=response_text,
        sources=retrieval.get_sources(),
        retrieval_count=retrieval.total_results,
        model_used=settings.claude_model,
    )


async def generate_response_stream(
    user_message: str,
    category: Optional[str] = None,
    top_k: int | None = None,
) -> AsyncIterator[str]:
    """
    Generate a streaming response.

    Args:
        user_message: The user's question
        category: Optional category filter
        top_k: Number of chunks to retrieve

    Yields:
        Response text chunks as they're generated
    """
    settings = get_settings()
    top_k = top_k or settings.default_top_k

    # Retrieve relevant context
    retrieval = retrieve(
        query=user_message,
        top_k=top_k,
        category=category,
    )

    logger.info(f"Retrieved {retrieval.total_results} chunks for query: {user_message[:50]}...")

    # Build prompt with context
    user_prompt = build_context_prompt(retrieval, user_message)

    # Generate streaming response with Claude
    client = get_async_anthropic_client()

    async with client.messages.stream(
        model=settings.claude_model,
        max_tokens=settings.max_response_tokens,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_prompt}
        ],
    ) as stream:
        async for text in stream.text_stream:
            yield text


def format_sources_markdown(sources: list[dict]) -> str:
    """
    Format sources as markdown for display.

    Args:
        sources: List of source dicts

    Returns:
        Markdown formatted string
    """
    if not sources:
        return ""

    lines = ["\n\n---\n\n**Sources:**"]
    for source in sources:
        title = source.get("title", "Unknown")
        category = source.get("category", "")
        duration = source.get("duration", "")
        url = source.get("vimeo_url", "")

        line = f"- **{title}**"
        if category:
            line += f" ({category})"
        if duration:
            line += f" [{duration}]"
        if url:
            line += f" - [Watch Video]({url})"

        lines.append(line)

    return "\n".join(lines)
