from dataclasses import dataclass
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from chunker import CodeChunk

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
@dataclass
class EmbeddedChunk:
    chunk: CodeChunk
    embedding: np.ndarray

class CodeEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
    def create_chunk_text(self, chunk: CodeChunk) -> str:
        return f"""
            Language: {chunk.language}
            Type: {chunk.chunk_type}
            Name: {chunk.name}
            File: {chunk.file_path}
            Code: 
            {chunk.code}
            """
    def embed_chunk(self, chunk: CodeChunk) -> EmbeddedChunk:
        text = self.create_chunk_text(chunk)
        embedding = self.model.encode(text, normalize_embeddings=True)
        return EmbeddedChunk(chunk=chunk, embedding=embedding)

    def embed_chunks(self, chunks: List[CodeChunk]) -> List[EmbeddedChunk]:
        texts = [
            self.create_chunk_text(chunk) for chunk in chunks
        ]
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        results = []
        for chunk, embedding in zip(chunks,embeddings):
            results.append(
                EmbeddedChunk(
                    chunk=chunk,
                    embedding=embedding
                )
            )
        return results