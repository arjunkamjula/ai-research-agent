"""
app/tools/document_lookup.py

ChromaDB semantic search tool.
Embeds the query and finds the most similar documents
in the persistent local vector store.
"""

import os
from functools import lru_cache

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR     = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "research_docs"


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(
        path     = CHROMA_DIR,
        settings = Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name     = COLLECTION_NAME,
        metadata = {"hnsw:space": "cosine"},
    )


@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def document_lookup(query: str, top_k: int = 4) -> str:
    """
    Search the vector store for documents similar to the query.

    Args:
        query: Natural language search query
        top_k: Number of results to return

    Returns:
        Formatted string with matching document chunks
    """
    try:
        collection = get_collection()
        model      = get_model()

        if collection.count() == 0:
            return "Vector store is empty. No documents have been ingested yet."

        query_embedding = model.encode(query, normalize_embeddings=True).tolist()

        results = collection.query(
            query_embeddings = [query_embedding],
            n_results        = min(top_k, collection.count()),
            include          = ["documents", "metadatas", "distances"],
        )

        docs      = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        if not docs:
            return f"No relevant documents found for: {query}"

        formatted = [f"Document search results for: {query}\n"]
        for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances), 1):
            similarity = round(1 - dist, 4)
            source     = meta.get("source", "unknown")
            formatted.append(
                f"[{i}] Source: {source} | Similarity: {similarity}\n"
                f"    {doc[:500]}{'...' if len(doc) > 500 else ''}\n"
            )

        return "\n".join(formatted)

    except Exception as e:
        return f"Document lookup failed: {e}"


def ingest_document(text: str, source: str = "manual") -> int:
    """
    Add a document to the vector store.
    Splits into chunks and embeds each one.

    Returns:
        Number of chunks added
    """
    collection = get_collection()
    model      = get_model()

    chunk_size = 500
    words      = text.split()
    chunks     = [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]
    chunks = [c for c in chunks if len(c) > 50]

    if not chunks:
        return 0

    embeddings = model.encode(chunks, normalize_embeddings=True).tolist()
    ids        = [f"{source}_{i}" for i in range(len(chunks))]
    metadatas  = [{"source": source, "chunk_index": i} for i in range(len(chunks))]

    collection.upsert(
        ids        = ids,
        documents  = chunks,
        embeddings = embeddings,
        metadatas  = metadatas,
    )

    return len(chunks)
