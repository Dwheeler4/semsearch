import os
from dataclasses import dataclass

SUPPORTED_EXTENSIONS = {".md", ".txt", ".py", ".cpp", ".c", ".h", ".hpp", ".rst", ".org"}
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


@dataclass
class Chunk:
    file: str
    start_line: int
    end_line: int
    text: str


def find_files(root: str) -> list[str]:
    matches = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in {"node_modules", "__pycache__", ".semsearch_index"}]
        for name in filenames:
            if os.path.splitext(name)[1] in SUPPORTED_EXTENSIONS:
                matches.append(os.path.join(dirpath, name))
    return matches


def chunk_text(lines: list[str], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[tuple[int, int, str]]:
    chunks = []
    start_idx = 0
    n = len(lines)
    while start_idx < n:
        text = ""
        end_idx = start_idx
        while end_idx < n and len(text) < chunk_size:
            text += lines[end_idx]
            end_idx += 1
        chunks.append((start_idx + 1, end_idx, text))
        if end_idx >= n:
            break
        overlap_chars = 0
        back_idx = end_idx
        while back_idx > start_idx and overlap_chars < overlap:
            back_idx -= 1
            overlap_chars += len(lines[back_idx])
        start_idx = max(back_idx, start_idx + 1)
    return chunks


def ingest_file(path: str) -> list[Chunk]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except OSError:
        return []
    if not lines:
        return []
    return [
        Chunk(file=path, start_line=start, end_line=end, text=text)
        for start, end, text in chunk_text(lines)
        if text.strip()
    ]


def ingest_directory(root: str) -> list[Chunk]:
    chunks = []
    for path in find_files(root):
        chunks.extend(ingest_file(path))
    return chunks
