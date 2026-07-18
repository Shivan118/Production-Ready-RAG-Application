# Retrieval-Augmented Generation (RAG) — Core Concepts

## What is RAG?

Retrieval-Augmented Generation (RAG) is an architecture that combines a
retrieval system with a large language model. Instead of relying only on the
model's training data, RAG retrieves relevant documents from a knowledge base
at query time and passes them to the LLM as context. This grounds answers in
real data, reduces hallucinations, and lets the system answer questions about
private or recent information.

## Chunking

Documents are split into smaller pieces called chunks before indexing.
Recursive character splitting keeps paragraphs and sentences intact where
possible. Typical chunk sizes range from 500 to 1500 characters with an
overlap of 10–20% so context is not lost at chunk boundaries.

## Embeddings

Each chunk is converted into a dense vector using an embedding model such as
OpenAI's text-embedding-3-small. Semantically similar texts map to nearby
vectors, which enables similarity search.

## Vector Databases

A vector database like ChromaDB stores embeddings and supports fast
approximate nearest-neighbor search. At query time the question is embedded
and the closest chunks are returned as candidate context.

## Hybrid Search

Hybrid search combines dense (semantic) retrieval with sparse keyword
retrieval such as BM25. Dense search captures meaning; keyword search catches
exact terms like product names, error codes, or acronyms. Results are fused,
commonly with Reciprocal Rank Fusion (RRF).

## Reranking

A reranker is a cross-encoder model that scores each candidate chunk against
the query with full attention over both texts. It is slower than vector search
but far more accurate, so it is applied to a small candidate set (for example
the top 20) to select the best few chunks.

## Evaluation

RAG systems are evaluated on metrics such as faithfulness (is the answer
grounded in the context?), answer relevancy, context precision, and context
recall. Frameworks like RAGAS automate these measurements using LLM judges.
