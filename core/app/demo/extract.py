"""Text out of an uploaded knowledge-base document (docs/gtm-demo-links.md §5.3).

Supported: PDF (brochures, price sheets), HTML (a saved web page), plain text and
Markdown. The type is decided from the file's own bytes, not trusted from its name, and
HTML is reduced to its visible text (scripts, styles and markup dropped). DOCX is not
supported yet: save as PDF.
"""
import io
import re
from html.parser import HTMLParser

SUPPORTED = "PDF, HTML, TXT or Markdown"


class ExtractionError(ValueError):
    pass


class _VisibleText(HTMLParser):
    _SKIP = {"script", "style", "noscript", "template", "svg", "head"}
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section",
              "article", "table", "ul", "ol"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self.skip += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self.skip:
            self.skip -= 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def _tidy(text: str) -> str:
    lines = [re.sub(r"[ \t ]+", " ", ln).strip() for ln in text.splitlines()]
    out, blank = [], False
    for ln in lines:
        if ln:
            out.append(ln)
            blank = False
        elif not blank and out:
            out.append("")
            blank = True
    return "\n".join(out).strip()


def detect_kind(data: bytes) -> str:
    head = data[:1024].lstrip()
    if data[:5] == b"%PDF-":
        return "pdf"
    if data[:4] == b"PK\x03\x04":
        return "zip"                       # DOCX/XLSX/ZIP: not a text document
    lowered = head[:512].lower()
    if lowered.startswith(b"<!doctype html") or b"<html" in lowered or b"<body" in lowered:
        return "html"
    return "text"


def extract_text(data: bytes, filename: str = "") -> tuple[str, str]:
    """(text, kind). Raises ExtractionError with a message an admin can act on."""
    if not data:
        raise ExtractionError("the file is empty")
    kind = detect_kind(data)
    if kind == "zip":
        raise ExtractionError(f"{filename or 'this file'} looks like a Word/Excel/ZIP file; "
                              f"upload {SUPPORTED} (save Word documents as PDF)")
    if kind == "pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise ExtractionError("the PDF is password-protected")
            text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        except ExtractionError:
            raise
        except Exception as e:   # noqa: BLE001 — any parser failure is one clear message
            raise ExtractionError(f"the PDF could not be read ({type(e).__name__})") from None
        text = _tidy(text)
        if len(text) < 20:
            raise ExtractionError("no text found in the PDF (is it a scanned image? "
                                  "OCR is not supported for demo uploads)")
        return text, kind
    try:
        decoded = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ExtractionError(f"the file is not UTF-8 text; upload {SUPPORTED}") from None
    if kind == "html":
        parser = _VisibleText()
        parser.feed(decoded)
        decoded = "".join(parser.parts)
    text = _tidy(decoded)
    if not text:
        raise ExtractionError("no text found in the file")
    return text, kind
