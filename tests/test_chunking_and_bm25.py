from regrag.bm25 import BM25Index, tokenize
from regrag.chunking import chunk_paragraphs
from regrag.ecfr import Paragraph


def para(section, label, text, group=None, is_table=False):
    return Paragraph(
        doc="Regulation Z", title=12, part=1026, section_id=section, section_title=f"§ {section} Title",
        kind="regulation", label=label, group=group or section, text=text, is_table=is_table,
        heading_path=["Regulation Z", f"§ {section} Title"], url=f"https://x/{section}",
    )


def test_chunks_never_cross_sections_and_merge_small_groups():
    ps = [para("1026.1", "§ 1026.1(a)", "short one", "1026.1(a)"),
          para("1026.1", "§ 1026.1(b)", "short two", "1026.1(b)"),
          para("1026.2", "§ 1026.2(a)", "text from another section")]
    chunks = chunk_paragraphs(ps)
    assert [c.section_id for c in chunks] == ["1026.1", "1026.2"]
    assert chunks[0].labels == ["§ 1026.1(a)", "§ 1026.1(b)"]
    assert "§ 1026.1(a) – 1026.1(b)" in chunks[0].breadcrumb


def test_oversized_groups_split_at_paragraph_boundaries():
    words = "word " * 120
    ps = [para("1026.5", f"§ 1026.5(a)({i})", words, "1026.5(a)") for i in range(1, 6)]
    chunks = chunk_paragraphs(ps, max_words=300)
    assert len(chunks) == 3
    assert all(len(c.body.split()) <= 300 for c in chunks)
    assert sum(len(c.blocks) for c in chunks) == 5


def test_tables_are_atomic_blocks():
    ps = [para("1026.9", "§ 1026.9(a)", "intro " * 10, "1026.9(a)"),
          para("1026.9", "§ 1026.9(a) (table)", "| a | b |\n|---|---|\n| 1 | 2 |", "1026.9(a)", True),
          para("1026.9", "§ 1026.9(b)", "after " * 10, "1026.9(b)")]
    chunks = chunk_paragraphs(ps)
    table_chunks = [c for c in chunks if any(b.label.endswith("(table)") for b in c.blocks)]
    assert len(table_chunks) == 1 and len(table_chunks[0].blocks) == 1


def test_tokenizer_keeps_citations_and_aliases():
    toks = tokenize("Under Reg Z § 1026.54(a)(1) and comment 52(b)(2)(i)-1")
    assert {"1026.54", "1026.54(a)", "1026.54(a)(1)", "c52(b)(2)(i)-1", "regulation_z"} <= set(toks)


def test_bm25_exact_citation_ranks_first():
    idx = BM25Index(
        ["a", "b", "c"],
        ["§ 1026.54(a) grace period finance charges", "§ 1026.55(b) rate increases",
         "§ 1026.5(b) periodic statement timing grace period"],
    )
    assert idx.search("what does 1026.54 say", 3)[0][0] == "a"
