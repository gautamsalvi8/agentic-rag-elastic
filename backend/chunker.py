"""
chunker.py
---------------------------------
Splits text into overlapping chunks
with IDs for proper indexing + retrieval
"""

import re


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size=300, overlap=50):
    text = clean_text(text)

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # avoid cutting mid-word
        if end < len(text):
            last_space = chunk.rfind(" ")
            if last_space != -1:
                chunk = chunk[:last_space]

        chunks.append({
            "text": chunk.strip(),
            "chunk_id": chunk_id
        })

        chunk_id += 1
        start += chunk_size - overlap

    return chunks
