"""Tests for file discovery and character-count chunking (semsearch/ingest.py)."""
import os

from semsearch.ingest import CHUNK_SIZE, chunk_text, find_files, ingest_file


def _lines(n: int, width: int = 40) -> list[str]:
    return [f"{'x' * width} line {i}\n" for i in range(n)]


def test_chunk_text_covers_every_line_with_no_gap():
    lines = _lines(200)
    chunks = chunk_text(lines)

    assert chunks[0][0] == 1
    assert chunks[-1][1] == len(lines)

    covered_through = 0
    for start, end, _ in chunks:
        assert 1 <= start <= end <= len(lines)
        assert start <= covered_through + 1  # nothing skipped between chunks
        covered_through = max(covered_through, end)
    assert covered_through == len(lines)


def test_chunk_text_respects_chunk_size_except_last_chunk():
    chunks = chunk_text(_lines(300))
    assert len(chunks) > 1
    for _, _, text in chunks[:-1]:
        assert len(text) >= CHUNK_SIZE


def test_chunk_text_overlaps_consecutive_chunks():
    chunks = chunk_text(_lines(300))
    for (_, end1, _), (start2, _, _) in zip(chunks, chunks[1:]):
        assert start2 <= end1


def test_chunk_text_makes_forward_progress_on_a_single_huge_line():
    chunks = chunk_text(["z" * (CHUNK_SIZE * 5) + "\n"] * 3)
    starts = [start for start, _, _ in chunks]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)  # strictly increasing -> no infinite loop


def test_chunk_text_on_empty_input():
    assert chunk_text([]) == []


def test_find_files_filters_by_extension_and_skips_noise(tmp_path):
    (tmp_path / "a.py").write_text("print(1)\n")
    (tmp_path / "b.md").write_text("# notes\n")
    (tmp_path / "c.json").write_text("{}\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "d.cpp").write_text("int main(){}\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "e.py").write_text("x = 1\n")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "f.py").write_text("x = 1\n")

    found = {os.path.relpath(p, tmp_path) for p in find_files(str(tmp_path))}
    assert found == {"a.py", "b.md", os.path.join("sub", "d.cpp")}


def test_ingest_file_handles_missing_and_empty_files(tmp_path):
    assert ingest_file(str(tmp_path / "does_not_exist.py")) == []
    empty = tmp_path / "empty.py"
    empty.write_text("")
    assert ingest_file(str(empty)) == []


def test_ingest_file_returns_chunks_carrying_path_and_line_numbers(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("".join(_lines(120)))

    chunks = ingest_file(str(f))
    assert chunks
    assert all(c.file == str(f) for c in chunks)
    assert all(c.text.strip() for c in chunks)
    assert chunks[0].start_line == 1
