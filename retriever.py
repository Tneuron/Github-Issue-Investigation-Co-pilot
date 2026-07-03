import pickle
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class HybridRetriever:
    def __init__(self, index_path="vector_store/code.index", metadata_path="vector_store/metadata.pkl", model_name="BAAI/bge-small-en-v1.5",):
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as file:
            self.metadata = pickle.load(file)
        self.model = SentenceTransformer(model_name)

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )
        return np.array([embedding], dtype=np.float32)

    def extract_symbols(self, query: str) -> list[str]:

        pattern = r"[A-Za-z_][A-Za-z0-9_]{2,}"
        symbols = re.findall(pattern, query)

        return list(dict.fromkeys(symbols))

    def tokenize(self, text: str) -> set[str]:
        return {
            token.lower()
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text)
        }

    def symbol_search(self, symbols: list[str]):
        results = []

        for chunk in self.metadata:
            score = 0.0

            name = chunk.name.lower()
            file_path = chunk.file_path.lower()
            code = chunk.code.lower()

            for symbol in symbols:
                token = symbol.lower()

                if token == name:
                    score += 30.0
                elif token in name:
                    score += 15.0
                elif token in file_path:
                    score += 10.0
                elif token in code:
                    score += 2.0

            if score > 0:
                results.append((score, chunk))

        return results

    def semantic_search(self, query: str, k: int = 20):
        query_vector = self.embed_query(query)
        distances, indices = self.index.search(query_vector, k)

        results = []

        for distance, index in zip(distances[0], indices[0]):
            if index < 0:
                continue

            results.append(
                (
                    -float(distance),
                    self.metadata[index],
                )
            )

        return results

    def rerank(self, query: str, results: list[dict]):
        """
        Apply generic lexical and chunk-type boosts after combining exact and
        semantic retrieval.
        """
        query_tokens = self.tokenize(query)

        for result in results:
            score = result["score"]
            chunk = result["chunk"]

            name_tokens = self.tokenize(chunk.name)
            path_tokens = self.tokenize(chunk.file_path)
            code_tokens = self.tokenize(chunk.code)

            for token in query_tokens:
                if token in name_tokens:
                    score += 8.0
                elif token in path_tokens:
                    score += 5.0
                elif token in code_tokens:
                    score += 0.5

            if chunk.chunk_type == "method":
                score += 2.0
            elif chunk.chunk_type == "function":
                score += 1.8
            elif chunk.chunk_type == "class":
                score += 0.5
            elif chunk.chunk_type == "imports":
                score -= 5.0

            result["score"] = score

        return sorted(
            results,
            key=lambda item: item["score"],
            reverse=True,
        )

    def search(self, query: str, k: int = 10):
        symbols = self.extract_symbols(query)

        symbol_results = self.symbol_search(symbols)
        semantic_results = self.semantic_search(
            query,
            k=max(k * 3, 20),
        )

        combined = {}

        for score, chunk in symbol_results:
            key = (chunk.file_path, chunk.name)

            combined[key] = {
                "score": score,
                "chunk": chunk,
            }

        for score, chunk in semantic_results:
            key = (chunk.file_path, chunk.name)

            if key in combined:
                combined[key]["score"] += score
            else:
                combined[key] = {
                    "score": score,
                    "chunk": chunk,
                }

        final_results = self.rerank(
            query,
            list(combined.values()),
        )

        return final_results[:k]