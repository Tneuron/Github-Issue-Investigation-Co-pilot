from file_parser import FileParser
from chunker import StructuralChunker
from embedder import CodeEmbedder
from indexer import FaissIndexer

REPO_PATH = "./flash-attention"

parser = FileParser(REPO_PATH)
documents = parser.parse_repository()
chunker = StructuralChunker()
print("Documents:", len(documents))
all_chunks = []
for doc in documents:
    all_chunks.extend(chunker.chunk_document(doc))
print("Total chunks:", len(all_chunks))

embedder = CodeEmbedder()
embedded_chunks = embedder.embed_chunks(all_chunks)
indexer = FaissIndexer()
indexer.save(embedded_chunks)

print(f"Chunks: {len(all_chunks)}")
for chunk in all_chunks:
    print()
    print("=" * 50)
    print(chunk.name)
    print(chunk.chunk_type)
    print(chunk.file_path)