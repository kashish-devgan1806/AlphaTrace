"""Offline tests for app/transcript.py's Q&A parsing -- no network/model,
pure text processing."""
from __future__ import annotations

from app.transcript import QASegment, parse_qa_segments, split_qa_section

TRANSCRIPT = """Prepared remarks go here, not part of the Q&A.

Questions and Answers

Q: Can you talk about gross margin trends?
A: We are seeing continued pressure, though we remain optimistic about
the back half of the year.

Q: What about receivables growth?
A: That growth was temporary and tied to one large customer.
"""


def test_split_qa_section_finds_header_case_insensitively():
    text = "Remarks.\n\nQUESTIONS AND ANSWERS\n\nQ: hi\nA: hello"
    section = split_qa_section(text)
    assert section.startswith("QUESTIONS AND ANSWERS")
    assert "Remarks." not in section


def test_split_qa_section_falls_back_to_whole_text_when_no_header():
    text = "Q: hi\nA: hello"
    assert split_qa_section(text) == text


def test_parse_qa_segments_happy_path():
    segments = parse_qa_segments(TRANSCRIPT)

    assert segments == [
        QASegment(
            segment_id=1,
            question="Can you talk about gross margin trends?",
            answer="We are seeing continued pressure, though we remain optimistic about the back half of the year.",
        ),
        QASegment(
            segment_id=2,
            question="What about receivables growth?",
            answer="That growth was temporary and tied to one large customer.",
        ),
    ]


def test_parse_qa_segments_ignores_text_before_header():
    segments = parse_qa_segments(TRANSCRIPT)
    for segment in segments:
        assert "Prepared remarks" not in segment.question
        assert "Prepared remarks" not in segment.answer


def test_parse_qa_segments_works_without_header():
    text = "Q: one question\nA: one answer"
    segments = parse_qa_segments(text)
    assert segments == [QASegment(segment_id=1, question="one question", answer="one answer")]


def test_parse_qa_segments_drops_trailing_unanswered_question():
    text = "Q: answered\nA: yes\nQ: never answered"
    segments = parse_qa_segments(text)
    assert len(segments) == 1
    assert segments[0].question == "answered"


def test_parse_qa_segments_ignores_orphan_answer_with_no_pending_question():
    text = "A: no question preceded this\nQ: real question\nA: real answer"
    segments = parse_qa_segments(text)
    assert segments == [QASegment(segment_id=1, question="real question", answer="real answer")]


def test_parse_qa_segments_empty_input_returns_empty_list():
    assert parse_qa_segments("") == []
    assert parse_qa_segments("no markers here at all") == []


def test_parse_qa_segments_new_question_overwrites_pending_one():
    text = "Q: first (superseded)\nQ: second\nA: answers the second"
    segments = parse_qa_segments(text)
    assert segments == [QASegment(segment_id=1, question="second", answer="answers the second")]
