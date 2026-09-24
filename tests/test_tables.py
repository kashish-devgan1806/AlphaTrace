"""Offline tests for app/tables.py — pure HTML parsing, no model/DB."""
from __future__ import annotations

from app.tables import extract_tables, strip_tables

FILING_HTML = """<html><body>
<div>Item 1A. Risk Factors</div>
<p>Some risk factor text describing risk in enough detail to be a real paragraph.</p>
<div>Item 8. Financial Statements</div>
<p>Consolidated statements follow.</p>
<table>
<tr><td>Metric</td><td>FY2025</td><td>FY2024</td></tr>
<tr><td>Revenue</td><td>100</td><td>90</td></tr>
<tr><td>Net Income</td><td>20</td><td>15</td></tr>
</table>
<p>End of statements.</p>
</body></html>"""


def test_extract_tables_finds_table_and_attributes_it_to_its_section():
    tables = extract_tables("doc-1", FILING_HTML, metadata={"ticker": "AAPL"})

    assert len(tables) == 1
    table = tables[0]
    assert table.doc_id == "doc-1"
    assert table.section == "Item 8. Financial Statements"
    assert table.rows == [
        ["Metric", "FY2025", "FY2024"],
        ["Revenue", "100", "90"],
        ["Net Income", "20", "15"],
    ]
    assert table.metadata == {"ticker": "AAPL"}


def test_extract_tables_serializes_to_markdown():
    tables = extract_tables("doc-1", FILING_HTML)

    assert tables[0].text == (
        "| Metric | FY2025 | FY2024 |\n"
        "| --- | --- | --- |\n"
        "| Revenue | 100 | 90 |\n"
        "| Net Income | 20 | 15 |"
    )


def test_extract_tables_skips_tiny_layout_tables():
    html = "<html><body><div>Item 1. Business</div><table><tr><td>x</td></tr></table></body></html>"

    assert extract_tables("doc-1", html) == []


def test_extract_tables_handles_no_tables():
    html = "<html><body><div>Item 1. Business</div><p>Just prose, no tables here.</p></body></html>"

    assert extract_tables("doc-1", html) == []


def test_extract_tables_pipe_in_cell_is_escaped():
    html = (
        "<html><body><div>Item 1. Business</div>"
        "<table><tr><td>A | B</td><td>1</td></tr><tr><td>C</td><td>2</td></tr></table>"
        "</body></html>"
    )

    tables = extract_tables("doc-1", html)

    assert "A \\| B" in tables[0].text


def test_extract_tables_ragged_rows_are_padded():
    html = (
        "<html><body><div>Item 1. Business</div>"
        "<table><tr><td>H1</td><td>H2</td><td>H3</td></tr>"
        "<tr><td>only one cell here but long enough</td></tr></table>"
        "</body></html>"
    )

    tables = extract_tables("doc-1", html)

    assert tables[0].rows == [["H1", "H2", "H3"], ["only one cell here but long enough"]]
    assert tables[0].text.splitlines()[2].count("|") == 4  # padded to 3 columns


def test_extract_tables_nested_table_folds_into_enclosing_cell():
    html = (
        "<html><body><div>Item 1. Business</div>"
        "<table><tr><td>Outer<table><tr><td>Inner</td></tr></table>text</td><td>1</td></tr>"
        "<tr><td>Row2</td><td>2</td></tr></table>"
        "</body></html>"
    )

    tables = extract_tables("doc-1", html)

    assert len(tables) == 1  # the nested <table> isn't captured as its own table
    assert tables[0].rows[0][0] == "OuterInnertext"


def test_extract_tables_toc_duplicate_table_keeps_the_real_occurrence():
    html = (
        "<html><body>"
        "<div>Item 8. Financial Statements</div>"  # TOC line
        "<div>Item 1A. Risk Factors</div><p>Risk text goes here in this paragraph.</p>"
        "<div>Item 8. Financial Statements</div>"  # real heading
        "<table><tr><td>Metric</td><td>Val</td></tr><tr><td>Revenue</td><td>1</td></tr></table>"
        "</body></html>"
    )

    tables = extract_tables("doc-1", html)

    assert len(tables) == 1
    assert tables[0].section == "Item 8. Financial Statements"


def test_strip_tables_removes_table_but_keeps_surrounding_prose():
    html = "<html><body><p>Before.</p><table><tr><td>a</td></tr></table><p>After.</p></body></html>"

    result = strip_tables(html)

    assert "<table>" not in result
    assert "<td>" not in result
    assert "<p>Before.</p>" in result
    assert "<p>After.</p>" in result


def test_strip_tables_removes_nested_tables_as_one_unit():
    html = "<div><table><tr><td>outer<table><tr><td>inner</td></tr></table></td></tr></table><p>Tail.</p></div>"

    result = strip_tables(html)

    assert "<table>" not in result
    assert "<p>Tail.</p>" in result


def test_strip_tables_no_tables_is_unchanged():
    html = "<html><body><p>Just prose.</p></body></html>"

    assert strip_tables(html) == html
