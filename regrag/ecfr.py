"""Download eCFR XML and parse it into labelled paragraphs.

eCFR section text is a flat list of <P> elements whose hierarchy lives only in
leading markers: ``(a)`` → ``(1)`` → ``(i)`` → ``(A)`` → ``(<I>1</I>)``. We
rebuild that hierarchy so every paragraph carries its full citation, e.g.
``1026.54(a)(1)(ii)``. Supplement I (Official Interpretations) is keyed by
headings like ``54(a)(1)`` plus numbered comments, which become citations such
as ``Comment 54(a)(1)-2``.
"""

from __future__ import annotations

import gzip
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

# (title, part, short name, long name)
SOURCES: list[tuple[int, int, str, str]] = [
    (12, 1026, "Regulation Z", "Truth in Lending"),
    (12, 1005, "Regulation E", "Electronic Fund Transfers"),
    (31, 1020, "31 CFR 1020", "BSA Rules for Banks (incl. CIP)"),
    (31, 1010, "31 CFR 1010", "BSA General Provisions (incl. Beneficial Ownership)"),
]

ECFR_URL = "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-{title}.xml?part={part}"


@dataclass
class Paragraph:
    doc: str  # "Regulation Z"
    title: int
    part: int
    section_id: str  # "1026.54" or "Supplement I" / "Appendix G"
    section_title: str
    kind: str  # "regulation" | "interpretation" | "appendix"
    label: str  # citable label: "§ 1026.54(a)(1)" / "Comment 54(a)(1)-1"
    group: str  # chunking group key within the section
    text: str
    is_table: bool = False
    heading_path: list[str] = field(default_factory=list)
    url: str = ""


def download(data_dir: str | Path, date: str) -> list[Path]:
    out_dir = Path(data_dir) / "ecfr"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for title, part, _, _ in SOURCES:
        path = out_dir / f"title{title}_part{part}.xml"
        if not path.exists():
            req = urllib.request.Request(
                ECFR_URL.format(date=date, title=title, part=part),
                headers={"Accept-Encoding": "gzip", "User-Agent": "regrag/1.0"},
            )
            raw = urllib.request.urlopen(req, timeout=120).read()
            try:
                raw = gzip.decompress(raw)
            except OSError:
                pass
            path.write_bytes(raw)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")
ITALIC_OPEN, ITALIC_CLOSE = "⟨", "⟩"  # marks italic paragraph markers


def _text(el, mark_italic: bool = False) -> str:
    parts: list[str] = []

    def walk(e):
        if e.text:
            parts.append(e.text)
        for c in e:
            if c.tag in ("FTREF",):
                pass
            elif c.tag == "I" and mark_italic:
                parts.append(ITALIC_OPEN)
                walk(c)
                parts.append(ITALIC_CLOSE)
            elif c.tag == "br":
                parts.append(" ")
            else:
                walk(c)
            if c.tail:
                parts.append(c.tail)

    walk(el)
    return _WS.sub(" ", "".join(parts)).strip()


def table_to_markdown(table) -> str:
    rows = []
    for tr in table.iter("TR"):
        cells = [_text(td).replace("|", "/") for td in tr if td.tag in ("TD", "TH")]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Paragraph marker hierarchy
# ---------------------------------------------------------------------------

_ROMAN = [
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
    "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx",
    "xxi", "xxii", "xxiii", "xxiv", "xxv",
]
_LEAD = re.compile(rf"^\(({ITALIC_OPEN}?)([A-Za-z0-9]+)({ITALIC_CLOSE}?)\)\s*")
_INLINE = re.compile(rf"^(.{{0,250}}?[—.{ITALIC_CLOSE}])\s*\(({ITALIC_OPEN}?)([A-Za-z0-9]+)({ITALIC_CLOSE}?)\)\s*")


def _letter_seq(m: str) -> int:
    return ord(m) - ord("a") + 1 if len(m) == 1 else 26 + (ord(m[0]) - ord("a") + 1)


class MarkerTracker:
    """Track the current (a)(1)(i)(A)(1)(i) path while walking a section."""

    def __init__(self) -> None:
        self.path: list[str] = []  # rendered markers, one per level
        self.raw: list[str] = []

    def _candidates(self, m: str, italic: bool) -> list[int]:
        """Return possible levels (0-based) for marker *m*."""
        if italic:
            return [4] if m.isdigit() else [5]
        if m.isdigit():
            return [1]
        if m.isupper():
            return [3]
        cands = []
        if m in _ROMAN:
            cands.append(2)
        if re.fullmatch(r"[a-z]|([a-z])\1", m):
            cands.append(0)
        return cands

    def _fits(self, level: int, m: str) -> bool:
        """Is *m* a plausible next marker at *level* given current state?"""
        if level > len(self.raw):
            return False
        prev = self.raw[level] if level < len(self.raw) else None
        if level == 0:
            return prev is None and m == "a" or prev is not None and _letter_seq(m) == _letter_seq(prev) + 1
        if level == 2:
            return (prev is None and m == "i") or (prev in _ROMAN and _ROMAN.index(prev) + 1 < len(_ROMAN) and _ROMAN[_ROMAN.index(prev) + 1] == m)
        return True

    def push(self, m: str, italic: bool) -> None:
        cands = self._candidates(m, italic)
        if len(cands) > 1:
            fitting = [c for c in cands if self._fits(c, m)]
            # A roman numeral that fits under a current (1)-level wins over a letter.
            cands = sorted(fitting or cands, reverse=True)
        level = cands[0] if cands else len(self.raw)
        level = min(level, len(self.raw))  # never skip levels
        self.raw = self.raw[:level] + [m]
        rendered = f"({m})"
        self.path = self.path[:level] + [rendered]

    def label(self) -> str:
        return "".join(self.path)


def _parse_markers(text: str, tracker: MarkerTracker) -> str:
    """Consume leading markers from *text*, update *tracker*, return clean text."""
    m = _LEAD.match(text)
    if not m:
        return text.replace(ITALIC_OPEN, "").replace(ITALIC_CLOSE, "")
    tracker.push(m.group(2), bool(m.group(1)))
    rest = text[m.end():]
    # Nested first-of-kind markers: "(3)(i) Application means…" or
    # "(a) Heading—(1) General rule." (second marker after the heading).
    while True:
        im = _LEAD.match(rest)
        italic, nxt = (im.group(1), im.group(2)) if im else (None, None)
        if not im:
            im = _INLINE.match(rest)
            italic, nxt = (im.group(2), im.group(3)) if im else (None, None)
        if not im or nxt not in ("1", "i", "A"):
            break
        tracker.push(nxt, bool(italic))
        rest = rest[im.end():]
    # The full text (heading words included) is kept; only italic sentinels are stripped.
    return text.replace(ITALIC_OPEN, "").replace(ITALIC_CLOSE, "")


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def _section_url(title: int, part: int, section_id: str, anchor: str = "") -> str:
    base = f"https://www.ecfr.gov/current/title-{title}/part-{part}/section-{section_id}"
    return f"{base}#p-{section_id}{anchor}" if anchor else base


def _appendix_url(title: int, part: int, name: str) -> str:
    slug = name.replace(" ", "%20")
    return f"https://www.ecfr.gov/current/title-{title}/part-{part}/appendix-{slug}"


def parse_section(div, doc: str, title: int, part: int) -> list[Paragraph]:
    section_id = div.get("N")
    head = div.find("HEAD")
    section_title = _text(head) if head is not None else section_id
    tracker = MarkerTracker()
    out: list[Paragraph] = []
    for child in div:
        if child.tag in ("P", "FP", "FP-1", "FP-2"):
            raw = _text(child, mark_italic=True)
            if not raw:
                continue
            text = _parse_markers(raw, tracker)
            anchor = tracker.label()
            top = tracker.path[0] if tracker.path else ""
            out.append(Paragraph(
                doc=doc, title=title, part=part, section_id=section_id,
                section_title=section_title, kind="regulation",
                label=f"§ {section_id}{anchor}", group=f"{section_id}{top}", text=text,
                heading_path=[doc, section_title], url=_section_url(title, part, section_id, anchor),
            ))
        elif child.tag in ("GPOTABLE", "TABLE") or child.find(".//TABLE") is not None:
            tables = [child] if child.tag == "TABLE" else list(child.iter("TABLE"))
            for t in tables:
                md = table_to_markdown(t)
                if md:
                    anchor = tracker.label()
                    top = tracker.path[0] if tracker.path else ""
                    out.append(Paragraph(
                        doc=doc, title=title, part=part, section_id=section_id,
                        section_title=section_title, kind="regulation",
                        label=f"§ {section_id}{anchor} (table)", group=f"{section_id}{top}",
                        text=md, is_table=True, heading_path=[doc, section_title],
                        url=_section_url(title, part, section_id, anchor),
                    ))
    return out


_INTERP_SECTION = re.compile(r"^Section (\d+\.\d+)")
_INTERP_ANCHOR = re.compile(r"^(?:Paragraphs? )?(\d+(?:\([A-Za-z0-9]+\))+)")
_COMMENT_NUM = re.compile(r"^(\d+)\.\s")
_SUB_ROMAN = re.compile(r"^([ivx]+)\.\s")
_SUB_UPPER = re.compile(r"^([A-Z])\.\s")


def parse_appendix(div, doc: str, title: int, part: int) -> list[Paragraph]:
    """Appendices and Supplement I. Headings drive grouping."""
    name = div.get("N")
    head = div.find("HEAD")
    app_title = _text(head) if head is not None else name
    is_interp = name.startswith("Supplement")
    url = _appendix_url(title, part, name)
    out: list[Paragraph] = []
    hd1 = hd2 = hd3 = ""
    interp_section = sec_heading = anchor_heading = ""
    anchor = ""
    comment = sub1 = sub2 = ""
    group_n = 0

    def label() -> str:
        if is_interp and anchor:
            if not comment:
                return f"Comment {anchor}"
            subs = "".join(f".{s}" for s in (sub1, sub2) if s)
            return f"Comment {anchor}-{comment}{subs}"
        if is_interp and interp_section:
            num = interp_section.split(".")[1]
            return f"Comment {num}-{comment}" if comment else f"Official Interpretation, § {interp_section}"
        return name + (f" — {hd3 or hd2 or hd1}" if (hd3 or hd2 or hd1) else "")

    def headings() -> list[str]:
        if is_interp:  # subpart headings add noise; keep section + paragraph heading
            return [doc, "Official Interpretations"] + [h for h in (sec_heading, anchor_heading) if h]
        return [doc, app_title] + [h for h in (hd1, hd2, hd3) if h]

    for child in div.iter():
        tag = child.tag
        if tag in ("HD1", "HD2", "HD3", "HED"):
            t = _text(child)
            if not t:
                continue
            group_n += 1
            if tag == "HD1":
                hd1, hd2, hd3 = t, "", ""
            elif tag == "HD2":
                hd2, hd3 = t, ""
            else:
                hd3 = t
            if is_interp:
                # Reg Z: "Section 1026.54—…" at HD2, anchors "54(a)(1) …" at HD3.
                # Reg E: "Section 1005.6 …" at HD1, anchors at HD2/HD3, some "Paragraph 2(b)(3)".
                if sm := _INTERP_SECTION.match(t):
                    interp_section, sec_heading = sm.group(1), t
                    anchor = anchor_heading = comment = ""
                elif am := _INTERP_ANCHOR.match(t):
                    anchor, anchor_heading, comment = am.group(1), t, ""
                elif tag in ("HD1", "HD2"):
                    interp_section = sec_heading = anchor = anchor_heading = comment = ""
        elif tag in ("P", "FP", "FP-1", "FP-2", "FP-DASH"):
            if child.getparent() is not None and child.getparent().tag in ("TD", "TH"):
                continue
            t = _text(child)
            if not t:
                continue
            if is_interp:
                if cm := _COMMENT_NUM.match(t):
                    comment, sub1, sub2 = cm.group(1), "", ""
                elif sm := _SUB_ROMAN.match(t):
                    sub1, sub2 = sm.group(1), ""
                elif sm := _SUB_UPPER.match(t):
                    sub2 = sm.group(1)
            out.append(Paragraph(
                doc=doc, title=title, part=part,
                section_id=interp_section if is_interp and interp_section else name,
                section_title=app_title, kind="interpretation" if is_interp else "appendix",
                label=label(), group=f"{name}#{group_n}", text=t,
                heading_path=headings(),
                url=url,
            ))
        elif tag == "TABLE":
            md = table_to_markdown(child)
            if md:
                out.append(Paragraph(
                    doc=doc, title=title, part=part, section_id=name, section_title=app_title,
                    kind="interpretation" if is_interp else "appendix",
                    label=f"{label()} (table)", group=f"{name}#{group_n}", text=md, is_table=True,
                    heading_path=headings(), url=url,
                ))
    return out


def parse_part(path: str | Path, doc: str, title: int, part: int) -> list[Paragraph]:
    root = etree.parse(str(path)).getroot()
    out: list[Paragraph] = []
    for div in root.iter("DIV8", "DIV9"):
        if div.get("TYPE") == "SECTION":
            out.extend(parse_section(div, doc, title, part))
        elif div.get("TYPE") == "APPENDIX":
            out.extend(parse_appendix(div, doc, title, part))
    return out


def load_corpus(data_dir: str | Path, date: str) -> list[Paragraph]:
    paths = download(data_dir, date)
    paragraphs: list[Paragraph] = []
    for path, (title, part, doc, _long) in zip(paths, SOURCES, strict=False):
        paragraphs.extend(parse_part(path, doc, title, part))
    return paragraphs
