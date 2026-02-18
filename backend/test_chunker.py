from chunker import chunk_text

sample = "Elasticsearch is powerful. " * 200

chunks = chunk_text(sample)

print("Chunks created:", len(chunks))
print(chunks[0])
