from lxml import etree

from regrag.ecfr import parse_appendix, parse_section, table_to_markdown


def _section(body: str):
    return etree.fromstring(f'<DIV8 N="1026.54" TYPE="SECTION"><HEAD>§ 1026.54 Limits.</HEAD>{body}</DIV8>')


def labels(paras):
    return [p.label for p in paras]


def test_marker_hierarchy_and_inline_markers():
    div = _section(
        "<P>(a) Heading—(1) General rule. Text.</P>"
        "<P>(i) First.</P><P>(ii) Second.</P>"
        "<P>(2) Definition.</P>"
        "<P>(b) Exceptions.</P><P>(1) One.</P>"
    )
    assert labels(parse_section(div, "Regulation Z", 12, 1026)) == [
        "§ 1026.54(a)(1)", "§ 1026.54(a)(1)(i)", "§ 1026.54(a)(1)(ii)",
        "§ 1026.54(a)(2)", "§ 1026.54(b)", "§ 1026.54(b)(1)",
    ]


def test_back_to_back_and_italic_heading_markers():
    div = _section(
        "<P>(a) Defs.</P>"
        "<P>(3)(i) Application means.</P><P>(ii) Other.</P>"
        "<P>(11) <I>Due date.</I> (i) Except as provided.</P><P>(A) The due date.</P>"
    )
    assert labels(parse_section(div, "Regulation Z", 12, 1026)) == [
        "§ 1026.54(a)", "§ 1026.54(a)(3)(i)", "§ 1026.54(a)(3)(ii)",
        "§ 1026.54(a)(11)(i)", "§ 1026.54(a)(11)(i)(A)",
    ]


def test_letter_i_vs_roman_i():
    # (i) after (h) at top level is the letter; (i) under (1) is roman.
    body = "".join(f"<P>({c}) X.</P>" for c in "abcdefgh") + "<P>(i) Letter i.</P><P>(1) Sub.</P><P>(i) Roman.</P>"
    got = labels(parse_section(_section(body), "Regulation Z", 12, 1026))
    assert got[-3:] == ["§ 1026.54(i)", "§ 1026.54(i)(1)", "§ 1026.54(i)(1)(i)"]


def test_interpretation_labels_reg_z_and_reg_e_styles():
    regz = etree.fromstring(
        '<DIV9 N="Supplement I to Part 1026" TYPE="APPENDIX"><HEAD>Supplement I</HEAD>'
        "<HD2>Section 1026.54—Limitations</HD2><HD3>54(a)(1) General Rule</HD3>"
        "<P>1. Eligibility. Text.</P><P>i. Example.</P><P>A. Sub.</P><P>2. Definition.</P></DIV9>"
    )
    p = parse_appendix(regz, "Regulation Z", 12, 1026)
    assert labels(p) == ["Comment 54(a)(1)-1", "Comment 54(a)(1)-1.i", "Comment 54(a)(1)-1.i.A",
                         "Comment 54(a)(1)-2"]
    assert all(x.section_id == "1026.54" and x.kind == "interpretation" for x in p)

    rege = etree.fromstring(
        '<DIV9 N="Supplement I to Part 1005" TYPE="APPENDIX"><HEAD>Supplement I</HEAD>'
        "<HD1>Section 1005.2 Definitions</HD1><HD3>Paragraph 2(b)(3)</HD3><P>1. Text.</P>"
        "<HD1>Section 1005.6 Liability</HD1><HD2>6(b) Limitations</HD2><P>2. Negligence.</P></DIV9>"
    )
    assert labels(parse_appendix(rege, "Regulation E", 12, 1005)) == ["Comment 2(b)(3)-1", "Comment 6(b)-2"]


def test_table_to_markdown():
    t = etree.fromstring("<TABLE><TR><TH>Age</TH><TH>Years</TH></TR><TR><TD>62</TD><TD>23</TD></TR></TABLE>")
    assert table_to_markdown(t) == "| Age | Years |\n|---|---|\n| 62 | 23 |"
