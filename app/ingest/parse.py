"""Turn a saved FDA warning-letter page into metadata + an ordered list of text blocks.

Blocks are the unit everything downstream refers to: the LLM extractor cites observation
spans as block ranges, and chunks record which blocks they cover.
"""
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from bs4 import BeautifulSoup, Tag

SIGNOFF = re.compile(r"^(sincerely|respectfully)\b|^/s/$", re.IGNORECASE)
EMPHASIS = {"strong", "b", "u", "em", "i"}
NUMBERED = re.compile(r"^(\d{1,2})\.\s+\S")


@dataclass
class Block:
    idx: int
    kind: str  # heading | para | item | footnote
    text: str


@dataclass
class ParsedDoc:
    meta: dict
    blocks: list[Block]

    @property
    def body(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


def _meta(soup: BeautifulSoup, key: str) -> str | None:
    tag = soup.find("meta", attrs={"property": key}) or soup.find("meta", attrs={"name": key})
    return tag["content"].strip() if tag and tag.get("content") else None


def _dl_field(soup: BeautifulSoup, label: str) -> str | None:
    for dt in soup.select("article dl dt"):
        if dt.get_text(strip=True).rstrip(":").lower() == label.lower():
            dd = dt.find_next_sibling("dd")
            return dd.get_text(" ", strip=True) if dd else None
    return None


def _text(el: Tag) -> str:
    # Inline tags (redaction spans, <em> organism names) must not break the sentence.
    t = el.get_text(" ")
    t = re.sub(r"\s+", " ", t).strip()
    return re.sub(r"\s+([,.;:)\]])", r"\1", re.sub(r"([(\[])\s+", r"\1", t))


def _is_heading(p: Tag, text: str) -> bool:
    if len(text) > 120 or NUMBERED.match(text) or text.endswith((".", ":")):
        return False
    if "(b)(" in text:
        return False
    # Heading = every visible character sits inside an emphasis tag.
    plain = [s for s in p.find_all(string=True) if s.strip() and not any(
        a.name in EMPHASIS for a in s.parents if a is not p and p in a.parents)]
    return not plain


def _blocks(soup: BeautifulSoup) -> list[Block]:
    col = max(soup.select("article div.col-md-8"), key=lambda c: len(c.get_text(strip=True)))
    for junk in col.select("dl, sup, script, style"):
        junk.decompose()

    out: list[Block] = []
    in_footnotes = signed_off = False
    for el in col.find_all(["p", "li", "hr", "h2", "h3", "h4"]):
        if el.name == "hr":
            in_footnotes = signed_off  # a rule after the signature introduces footnotes
            continue
        if el.name == "p" and el.find_parent("li"):
            continue
        text = _text(el)
        if re.fullmatch(r"_{5,}", text):
            in_footnotes = signed_off
            continue
        if not text:
            continue
        if SIGNOFF.match(text):
            signed_off = True
            continue
        if signed_off and not in_footnotes:
            continue  # signature block / cc list: no regulatory content
        if in_footnotes:
            kind = "footnote"
        elif el.name in ("h2", "h3", "h4") or (el.name == "p" and _is_heading(el, text)):
            kind = "heading"
        else:
            kind = "item" if el.name == "li" else "para"
        out.append(Block(len(out), kind, text))
    return out


def parse(path: Path) -> ParsedDoc:
    soup = BeautifulSoup(path.read_text(), "lxml")
    title = _meta(soup, "dcterms.title") or soup.title.get_text(strip=True)
    # Title format: "<Company> - <CMS #> - <MM/DD/YYYY>"
    m = re.match(r"^(?P<company>.+?)\s+-\s+(?P<cms>\d+)\s+-\s+(?P<date>\d{2}/\d{2}/\d{4})$", title)
    issue_date: date | None = None
    if m:
        month, day, year = map(int, m["date"].split("/"))
        issue_date = date(year, month, day)
    meta = {
        "id": path.stem,
        "doc_type": "warning_letter",
        "url": _meta(soup, "og:url"),
        "title": title,
        "company": m["company"] if m else None,
        "cms_number": m["cms"] if m else None,
        "issue_date": issue_date,
        "subject": _meta(soup, "dcterms.description"),
        "product": _dl_field(soup, "Product"),
        "issuing_office": _meta(soup, "dcterms.creator"),
        "street": _meta(soup, "og:street_address"),
        "city": _meta(soup, "og:locality"),
        "region": _meta(soup, "og:region"),
        "postal_code": _meta(soup, "og:postal_code"),
        "country": _meta(soup, "og:country_name"),
    }
    return ParsedDoc(meta, _blocks(soup))
