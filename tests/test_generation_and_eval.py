from types import SimpleNamespace as NS

import pytest

from regrag.chunking import Block, Chunk
from regrag.config import Settings
from regrag.evaluation import label_matches, retrieval_metrics
from regrag.generation import Generator, build_search_results, parse_response
from regrag.retrieval import Hit, RetrievalResult, rrf


def chunk(cid, labels):
    return Chunk(id=cid, doc="Regulation E", section_id="1005.11", kind="regulation", title="t",
                 breadcrumb=f"bc {cid}", url=f"https://x/{cid}",
                 blocks=[Block(label=lbl, text=f"text of {lbl}") for lbl in labels])


def cite(idx, start, end, text="quoted"):
    return NS(type="search_result_location", search_result_index=idx, start_block_index=start,
              end_block_index=end, cited_text=text)


def test_rrf_rewards_agreement():
    fused = rrf([["a", "b", "c"], ["c", "a", "d"]], k=60)
    assert [d for d, _ in fused][:2] == ["a", "c"]


def test_search_results_one_block_per_paragraph():
    hits = [Hit(chunk("x", ["§ 1005.11(c)(1)", "§ 1005.11(c)(2)"]), 1.0)]
    sr = build_search_results(hits)
    assert sr[0]["type"] == "search_result" and sr[0]["citations"] == {"enabled": True}
    assert [c["text"].split("]")[0] for c in sr[0]["content"]] == ["[§ 1005.11(c)(1)", "[§ 1005.11(c)(2)"]


def test_parse_response_maps_citations_to_paragraph_labels_and_dedupes():
    hits = [Hit(chunk("x", ["§ 1005.11(c)(1)", "§ 1005.11(c)(2)", "§ 1005.11(c)(3)"]), 1.0)]
    content = [
        NS(type="thinking"),
        NS(type="text", text="Ten business days.", citations=[cite(0, 0, 1)]),
        NS(type="text", text=" Up to 45 days with provisional credit.", citations=[cite(0, 1, 3), cite(0, 0, 1)]),
        NS(type="text", text=" Bogus.", citations=[cite(7, 0, 1)]),  # out-of-range index is dropped
    ]
    segments, citations, plain = parse_response("q", hits, content)
    assert [c.labels for c in citations] == [["§ 1005.11(c)(1)"], ["§ 1005.11(c)(2)", "§ 1005.11(c)(3)"]]
    assert segments[1].citations == [2, 1]
    assert segments[2].citations == []
    assert plain.startswith("Ten business days.[1]")


def test_abstains_below_threshold_without_calling_llm():
    class Boom:
        def __getattr__(self, name):
            raise AssertionError("LLM must not be called")

    gen = Generator(Settings(abstain_threshold=0.0), client=Boom())
    res = RetrievalResult("q", "hybrid_rerank", [Hit(chunk("x", ["§ 1"]), -5.0, rerank_score=-5.0)])
    ans = gen.answer("q", res)
    assert ans.abstained and ans.citations == []


@pytest.mark.parametrize("label,gold,ok", [
    ("§ 1026.13(c)(2)", "§ 1026.13(c)", True),
    ("§ 1026.13(cc)", "§ 1026.13(c)", False),
    ("§ 1026.5(b)(2)(iii)", "§ 1026.5(b)(2)(ii)", False),
    ("Comment 6(b)-2.i", "Comment 6(b)-2", True),
    ("Comment 6(b)-20", "Comment 6(b)-2", False),
    ("§ 1026.50(a)", "§ 1026.5", False),
])
def test_label_matching_respects_paragraph_boundaries(label, gold, ok):
    assert label_matches(label, gold) is ok


def test_retrieval_metrics():
    ranked = [chunk("1", ["§ 9(a)"]), chunk("2", ["§ 1005.11(c)(1)"]), chunk("3", ["§ 1005.11(c)(2)"])]
    m = retrieval_metrics(ranked, ["§ 1005.11(c)(1)", "§ 1005.11(c)(2)"], ks=(1, 3))
    assert m["recall@1"] == 0 and m["recall@3"] == 1.0 and m["mrr"] == 0.5


def test_ensure_rule_text_swaps_in_regulation_from_bottom():
    from regrag.retrieval import Retriever

    def mk(cid, kind, section):
        c = chunk(cid, [f"{cid}-label"])
        c.kind, c.section_id = kind, section
        return c

    r = Retriever.__new__(Retriever)
    r.settings = Settings(abstain_threshold=0.0)
    r.chunks = {c.id: c for c in [
        mk("i1", "interpretation", "1026.19"), mk("i2", "interpretation", "1026.32"),
        mk("i3", "interpretation", "1026.19"), mk("r19", "regulation", "1026.19"),
        mk("r32", "regulation", "1026.32"), mk("x", "interpretation", "1026.5"),
    ]}
    ranked = [("i1", 9), ("i2", 8), ("i3", 7), ("x", 6), ("r19", 5), ("r32", 4)]
    top = r.ensure_rule_text(ranked, k=4)
    assert [d for d, _ in top] == ["i1", "i2", "r19", "r32"]


def test_cited_chunks_pins_explicit_citations():
    from regrag.retrieval import Retriever

    r = Retriever.__new__(Retriever)
    a, b = chunk("a", ["§ 1026.54(a)(1)", "§ 1026.54(b)"]), chunk("b", ["Comment 52(b)(2)(i)-1"])
    r.chunks = {"a": a, "b": b, "c": chunk("c", ["§ 1026.5(b)"])}
    assert r.cited_chunks("What does 12 CFR 1026.54(b) say?") == ["a"]
    assert r.cited_chunks("Summarize comment 52(b)(2)(i)-1.") == ["b"]
    assert r.cited_chunks("grace periods in general") == []
