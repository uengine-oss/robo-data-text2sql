from __future__ import annotations

from xml.sax.saxutils import escape as xml_escape


def to_cdata(value: str) -> str:
    return f"<![CDATA[{value}]]>"


def emit_text(tag: str, text: str) -> str:
    return f"<{tag}>{xml_escape(text or '')}</{tag}>"


__all__ = ["to_cdata", "emit_text"]


