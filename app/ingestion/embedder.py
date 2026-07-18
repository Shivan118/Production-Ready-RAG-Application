"""OpenAI embeddings — cached per effective API key."""

from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from app.config import get_settings
from app.runtime_keys import effective_openai_key


@lru_cache(maxsize=8)
def _embeddings_for(api_key: str) -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=api_key,
    )


def get_embeddings() -> OpenAIEmbeddings:
    return _embeddings_for(effective_openai_key())
