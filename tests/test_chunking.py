import pytest

from cited_rag.chunking import WhitespaceTokenizer, chunk_pages, windows
from cited_rag.pdf import clean_text, section_for_pages
from cited_rag.types import Page


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_windows_cover_everything_with_overlap():
    w = windows(23, 10, 2)
    assert w == [(0, 10), (8, 18), (16, 23)]
    covered = set()
    for s, e in w:
        covered.update(range(s, e))
    assert covered == set(range(23))


def test_windows_validation():
    with pytest.raises(ValueError):
        windows(10, 5, 5)
    with pytest.raises(ValueError):
        windows(10, 0, 0)
    assert windows(0, 5, 1) == []


def test_chunk_sizes_and_overlap():
    pages = [Page("d", 1, words(120), "S")]
    chunks, parents = chunk_pages(pages, WhitespaceTokenizer(), 50, 10, parent_size=0)
    toks = [c.text.split() for c in chunks]
    assert all(len(t) <= 50 for t in toks)
    assert toks[0][-10:] == toks[1][:10]
    assert parents == []
    assert all(c.parent_id == c.chunk_id for c in chunks)


def test_page_ranges_follow_source_pages():
    pages = [Page("d", 3, words(30, "a"), "S"), Page("d", 4, words(30, "b"), "S")]
    chunks, _ = chunk_pages(pages, WhitespaceTokenizer(), 40, 0, parent_size=0)
    assert (chunks[0].page_start, chunks[0].page_end) == (3, 4)
    assert (chunks[1].page_start, chunks[1].page_end) == (4, 4)
    assert chunks[0].cite() == "d, pp. 3-4"


def test_sections_split_blocks_and_parents_contain_children():
    pages = [
        Page("d", 1, words(30, "a"), "Intro"),
        Page("d", 2, words(100, "b"), "Body"),
    ]
    chunks, parents = chunk_pages(pages, WhitespaceTokenizer(), 20, 5, parent_size=60)
    assert {c.section for c in chunks} == {"Intro", "Body"}
    by_id = {p.parent_id: p for p in parents}
    for c in chunks:
        assert c.text in by_id[c.parent_id].text
        assert by_id[c.parent_id].section == c.section
    # Body is 100 tokens -> two parents of <=60 tokens
    assert sum(p.section == "Body" for p in parents) == 2


def test_clean_text_joins_hyphenation():
    assert clean_text("trust-\nworthy  systems\n\n\n\nok") == "trustworthy systems\n\nok"


def test_section_for_pages_uses_deepest_preceding_heading():
    toc = [[1, "Intro", 1], [1, "1.", 2], [2, "Methods", 2], [1, "End", 4]]
    assert section_for_pages(toc, 5) == ["Intro", "Methods", "Methods", "End", "End"]


def test_parent_page_breaks_map_offsets_to_pages():
    pages = [Page("d", 7, words(10, "a"), "S"), Page("d", 8, words(10, "b"), "S")]
    _, parents = chunk_pages(pages, WhitespaceTokenizer(), 5, 0, parent_size=100)
    (p,) = parents
    assert (p.page_start, p.page_end) == (7, 8)
    assert p.page_at(0) == 7
    assert p.page_at(p.text.index("b0")) == 8


def test_clean_text_drops_toc_leader_lines():
    assert clean_text("Contents\n1.2 Risk tolerance ........ 7\nBody text") == "Contents\nBody text"
    assert clean_text("3.1 Valid . . . . . . . . 13\nok") == "ok"
