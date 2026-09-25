"""Q&A transcript parsing for the Sentiment Agent.

A deliberately simple, documented input contract for a manually-supplied
transcript -- not a general parser for arbitrary vendor formats (that's a
real, separate problem, deferred the same way Agent 2 scoped its own
real-PDF-slide-deck gap to one synthetic file rather than blocking on a
real deck existing).

Expected shape:
  - An optional Q&A section header (a case-insensitive match against
    QA_HEADER_MARKERS, e.g. "Questions and Answers"); only text after it
    is parsed. If no marker is found, the whole text is treated as the
    Q&A section.
  - Turns marked with a line starting "Q:" (an analyst's question) or
    "A:" (management's answer), each continuing until the next marker.
    An unanswered trailing question (no "A:" line follows it) is dropped,
    not treated as an error -- a cut-off transcript is legitimate input.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

QA_HEADER_MARKERS = (
    "questions and answers",
    "question-and-answer session",
    "q&a session",
    "q&a",
)

_TURN_RE = re.compile(r"^(Q|A):\s?(.*)$")


@dataclass
class QASegment:
    """One analyst-question / management-answer pair. `segment_id` is
    1-based and reflects order of appearance in the transcript."""

    segment_id: int
    question: str
    answer: str


def split_qa_section(text: str) -> str:
    """Return the substring of `text` starting at its Q&A header, matched
    case-insensitively against QA_HEADER_MARKERS. Returns `text` unchanged
    if no marker is found -- a defensible degrade, not a hard requirement,
    since a manually-supplied transcript may already be Q&A-only."""
    lower = text.lower()
    earliest = None
    for marker in QA_HEADER_MARKERS:
        idx = lower.find(marker)
        if idx != -1 and (earliest is None or idx < earliest):
            earliest = idx
    return text[earliest:] if earliest is not None else text


def parse_qa_segments(text: str) -> list[QASegment]:
    """Parse the Q&A section of `text` into ordered QASegments. Walks
    "Q:"/"A:"-marked turns line by line, accumulating each turn's text
    until the next marker, and pairs each question with the answer that
    follows it. A new "Q:" while a question is already pending overwrites
    the pending one (only the most recent unanswered question is kept);
    an "A:" with no pending question is ignored (no orphan answers)."""
    section = split_qa_section(text)

    turns: list[tuple[str, str]] = []
    current_kind: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        if current_kind is not None:
            turns.append((current_kind, " ".join(current_lines).strip()))

    for line in section.splitlines():
        match = _TURN_RE.match(line.strip())
        if match:
            flush()
            current_kind, first = match.group(1), match.group(2)
            current_lines = [first] if first else []
        elif current_kind is not None:
            stripped = line.strip()
            if stripped:
                current_lines.append(stripped)
    flush()

    segments: list[QASegment] = []
    pending_question: str | None = None
    for kind, content in turns:
        if kind == "Q":
            pending_question = content
        elif kind == "A" and pending_question is not None:
            segments.append(QASegment(segment_id=len(segments) + 1, question=pending_question, answer=content))
            pending_question = None

    return segments
