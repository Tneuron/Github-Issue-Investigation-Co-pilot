from indexer import FaissIndexer
from retriever import HybridRetriever

indexer = FaissIndexer()
index, metadata = indexer.load()
print(f"Total vectors: {index.ntotal}")

retriever = HybridRetriever()
query = "How are users authenticated?"
results = retriever.search(query=query,k=5)
print(f"Query: {query}")
print("\nTop Results:\n")
for i, result in enumerate(results, start=1):
    chunk = result["chunk"]
    print("=" * 80)
    print(f"Rank: {i}")
    print(f"Score: "f"{result['score']:.4f}")
    print(f"Name: "f"{chunk.name}")
    print(f"Type: "f"{chunk.chunk_type}")
    print(f"Language: "f"{chunk.language}")
    print(f"File: "f"{chunk.file_path}")
    print("\nCode Preview:\n")
    print(chunk.code[:500])
    print()