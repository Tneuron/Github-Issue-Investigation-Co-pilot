from pathlib import Path
import pickle

import faiss
import numpy as np

from embedder import EmbeddedChunk

class FaissIndexer:
    def __init__(self, index_dir: str = "vector_store"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True,exist_ok=True)
        self.index_path = (self.index_dir / "code.index")
        self.metadata_path = (self.index_dir / "metadata.pkl")

    def build_index(self,embedded_chunks: list[EmbeddedChunk]):
        if not embedded_chunks:
            raise ValueError(
                "No embedded chunks provided."
            )
        vectors = np.array([chunk.embedding for chunk in embedded_chunks],dtype=np.float32)
        dimension = vectors.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)
        metadata = [chunk.chunk
            for chunk in embedded_chunks
        ]
        return index, metadata

    def save(self, embedded_chunks: list[EmbeddedChunk]):
        index, metadata = self.build_index(embedded_chunks)
        faiss.write_index(index, str(self.index_path))
        with open(self.metadata_path,"wb") as f:
            pickle.dump(metadata,f)
        print(f"Saved {len(metadata)} chunks")
        print(f"Index: {self.index_path}")
        print(f"Metadata: {self.metadata_path}")

    def load(self):
        if not self.index_path.exists():
            raise FileNotFoundError(self.index_path)
        if not self.metadata_path.exists():
            raise FileNotFoundError(self.metadata_path)
        index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path,"rb") as f:
            metadata = pickle.load(f)
        return index, metadata