from pathlib import Path
from dataclasses import dataclass
from typing import List

REPO_PATH = "./flash-attention"

SUPPORTED_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
}

IGNORED_DIRS = {
    ".git",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".venv",
    "venv",
}

@dataclass
class Document:
    file_path: str
    file_name: str
    extension: str
    content: str

class FileParser:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
    def parse_repository(self) -> List[Document]:
        documents = []
        for file_path in self.repo_path.rglob("*"):
            if not file_path.is_file():
                continue
            if any(part in IGNORED_DIRS for part in file_path.parts):
                continue
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            try:
                content = file_path.read_text(encoding="utf-8",errors="ignore")
                documents.append(
                    Document(
                        file_path=str(file_path.relative_to(self.repo_path)),
                        file_name=file_path.name,
                        extension=file_path.suffix,
                        content=content
                    )
                )
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")
        return documents

if __name__ == "__main__":
    parser = FileParser(REPO_PATH)
    documents = parser.parse_repository()
    print(f"\nParsed {len(documents)} files\n")
    for doc in documents[:5]:
        print("=" * 60)
        print(doc.file_path)
        print(f"Characters: {len(doc.content)}")