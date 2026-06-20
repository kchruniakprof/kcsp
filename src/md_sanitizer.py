"""
Deterministic markdown sanitizer for docling OCR output.
Must run BEFORE hierarchy_parser — ligature artifacts break header regex matching.
"""
import re


_LIGATURE_PATTERNS = [
    # OCR splits ligature glyphs with a space before the continuation letter.
    # Only merge when followed by a lowercase letter (safe: avoids merging sentence boundaries).
    (re.compile(r'fl ([a-zäöüß])', re.UNICODE), r'fl\1'),
    (re.compile(r'fi ([a-zäöüß])', re.UNICODE), r'fi\1'),
]

_IMAGE_PLACEHOLDER = re.compile(r'<!-- image -->')
_MULTI_BLANK = re.compile(r'\n{3,}')
_TRAILING_WS = re.compile(r'[ \t]+$', re.MULTILINE)


def sanitize(text: str) -> str:
    for pattern, replacement in _LIGATURE_PATTERNS:
        text = pattern.sub(replacement, text)

    text = _IMAGE_PLACEHOLDER.sub('', text)
    text = _MULTI_BLANK.sub('\n\n', text)
    text = _TRAILING_WS.sub('', text)

    return text
