class Prompts:

    # safest for RAG (grounded answers only)
    STRICT = """
You are a helpful assistant.
Answer ONLY using the context below.
If answer is not present, say: "Not found in documents".

Context:
{context}

Question:
{query}

Answer:
"""

    # more natural explanation
    EXPLAIN = """
You are a friendly assistant.
Explain the answer clearly in simple words using ONLY the context.

Context:
{context}

Question:
{query}

Answer:
"""

    # short + crisp
    CONCISE = """
Answer the question in 2-3 lines using the context only.

Context:
{context}

Question:
{query}

Answer:
"""
