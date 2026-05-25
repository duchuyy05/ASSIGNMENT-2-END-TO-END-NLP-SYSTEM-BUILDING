"""Build assignment data from manually curated QA CSV files.

This script treats data/manual_annotations/train_qa.csv and test_qa.csv as the
source of truth. It never modifies those CSV files.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import json
import random
import re
import subprocess
import tempfile
import time
import unicodedata
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MANUAL_DIR = DATA_DIR / "manual_annotations"
TRAIN_CSV = MANUAL_DIR / "train_qa.csv"
TEST_CSV = MANUAL_DIR / "test_qa.csv"
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATION_DIR = DATA_DIR / "annotations"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"


@dataclass
class SourceText:
    url: str
    title: str
    text: str
    source_type: str
    fetch_status: str
    error: str | None = None


class VisibleTextParser(HTMLParser):
    SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas"}
    BLOCK_TAGS = {
        "article",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self.skip_depth = 0
        self.in_title = False
        self.in_table_row = False
        self.in_table_cell = False
        self.current_table_row: list[str] = []
        self.current_table_cell: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
        if tag == "title":
            self.in_title = True
        if tag == "tr":
            self.in_table_row = True
            self.current_table_row = []
            return
        if tag in {"td", "th"} and self.in_table_row:
            self.in_table_cell = True
            self.current_table_cell = []
            return
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
        if tag == "title":
            self.in_title = False
        if tag in {"td", "th"} and self.in_table_cell:
            cell_text = normalize_text(" ".join(self.current_table_cell))
            if cell_text:
                self.current_table_row.append(cell_text)
            self.current_table_cell = []
            self.in_table_cell = False
            return
        if tag == "tr" and self.in_table_row:
            row_text = normalize_text(" | ".join(self.current_table_row))
            if row_text:
                self.parts.append("\n")
                self.parts.append(row_text)
                self.parts.append("\n")
            self.current_table_row = []
            self.in_table_row = False
            self.in_table_cell = False
            return
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        text = html.unescape(data).strip()
        if not text:
            return
        if self.in_title:
            self.title_parts.append(text)
        if self.in_table_cell:
            self.current_table_cell.append(text)
            return
        self.parts.append(text)
        self.parts.append(" ")

    @property
    def title(self) -> str:
        return normalize_text(" ".join(self.title_parts))

    @property
    def text(self) -> str:
        return normalize_text(" ".join(self.parts))


def normalize_text(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"(?m)^[ \t]*[✯★☆✭✰]\s*", "", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def clean_field(value: str) -> str:
    value = normalize_text(value)
    value = value.replace("\ufeff", "")
    return value


BOILERPLATE_TERMS = {
    "arrow_right_alt",
    "arrow right alt",
    "filter_alt",
    "filter alt",
    "cancel",
    "khoa hoc",
    "tinh diem xet hoc ba",
    "xem them",
    "cong cu chung",
    "vao lop 10",
    "cao dang",
    "xem de an cua cac truong khac",
    "tai file pdf de an",
    "diem thi tot nghiep thpt",
    "cach tinh diem xet tuyen",
    "diem uu tien",
    "diem khuyen khich",
    "quy doi diem ielts",
    "demo dang ky",
    "tra cuu xep hang",
    "dem nguoc",
    "tinh nang huu ich",
    "tra cuu de an tuyen sinh",
    "tra cuu diem chuan",
    "tra cuu to hop mon",
    "tra cuu tai",
    "co quan chu quan",
    "cong nghe giao duc thanh phat",
    "nhap ten truong",
    "click vao phuong thuc",
    "xem diem chuan dai hoc",
    "xem de an tuyen sinh",
    "cong cu tinh diem tot nghiep",
    "cong cu tinh diem hoc ba",
    "cac nganh nghe dao tao",
    "to hop xet tuyen dai hoc",
    "tel",
    "hotline",
    "van phong tang",
    "giay phep",
    "chiu trach nhiem noi dung",
    "cop - cong dong chuyen mon",
    "cop cong dong chuyen mon",
    "dang ky lich tuan",
    "tra cuu cham cong",
    "tra cuu lich giang day",
    "chien luoc phat trien",
    "co cau to chuc",
    "so do to chuc",
    "don vi dao tao",
    "y nghia logo",
    "bo nhan dien thuong hieu",
    "tin tuc su kien",
    "su kien sap dien ra",
    "pho bien phap luat",
    "thong bao tuyen sinh dao tao thac si",
    "thong bao tuyen sinh dao tao tien si",
    "lien ket",
    "he thong tac nghiep",
    "cong thong tin sinh vien",
    "cong thong tin can bo",
    "cong thong tin tuyen sinh",
    "he thong thong tin don vi",
    "thu vien dien tu",
    "he thong quan ly van ban",
    "hanh chinh mot cua",
    "tra cuu thi dua",
    "he thong ql khcn",
    "phan mem quan ly hoi thao",
    "vnu email",
    "quan ly dieu hanh",
    "ban tin noi bo",
    "su kien va thanh tuu noi bat",
    "quy dinh tai chinh",
    "huong dan thu hoc phi",
    "huong dan thanh toan",
    "theo doi tai san",
    "dam bao chat luong",
    "gioi thieu trung tam dbcl",
    "khao sat y kien",
    "quy dinh ve van thu luu tru",
    "ket luan cua ban giam hieu",
    "chuong trinh dao tao da kiem dinh",
    "thong tin khac ve dbcl",
    "to chuc doan the",
    "dang bo truong",
    "gioi thieu to chuc cong doan",
    "tong lien doan lao dong",
    "cong doan giao duc",
    "cong doan dai hoc quoc gia",
    "cong doan truong",
    "cong doan cac don vi",
    "tin hoat dong cong doan",
    "chuyen trang cong doan",
    "facebook ulis",
    "dao tao truc tuyen",
    "phong dao tao nguoi hoc",
    "khoa giao duc quoc te",
    "ulis success course",
    "unc ulis national conference",
    "de an ba vi",
    "chuong trinh tay bac",
    "dhhn hon 70 nam",
    "thpt chuyen ngoai ngu",
    "chuong trinh dao tao chuan clc",
    "chuong trinh dao tao thu hai",
    "chuong trinh dao tao van bang 2",
    "hoat dong hop tac",
    "chuyen trang dien dan",
    "tin tuc ve dhqghn",
    "co the ban quan tam",
    "phien ban website cu",
    "an pham gioi thieu",
    "video gioi thieu",
    "cac bai hat truyen thong",
    "fanpage",
    "copyright",
    "read more",
    "view more",
    "related posts",
    "you may be interested",
}

STANDALONE_HEADING_LINES = {
    "thong tin chinh",
    "phuong an tuyen sinh",
    "danh sach nganh dao tao",
    "diem chuan",
    "thoi gian va ho so xet tuyen",
    "hoc phi",
    "gioi thieu truong",
}


def ascii_fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in normalized if not unicodedata.combining(char))
    return text.replace("đ", "d").replace("Đ", "D").lower()


def line_key(line: str) -> str:
    return re.sub(r"\W+", "", ascii_fold(line))


def is_boilerplate_line(line: str, title: str | None = None) -> bool:
    lowered = line.lower().strip()
    folded = ascii_fold(line).strip()
    folded_words = re.sub(r"[^a-z0-9]+", " ", folded).strip()
    if not lowered:
        return True
    if title and line_key(line) == line_key(title):
        return True
    if folded_words in STANDALONE_HEADING_LINES:
        return True
    if lowered in {"menu", "home", "search", "email", "library", "english", "vietnamese"}:
        return True
    if re.match(r"^\d+\s*\.\s+", lowered):
        return True
    if line.count(".") >= 8:
        return True
    if "http://" in lowered or "https://" in lowered:
        return True
    if re.search(r"\d{4}-\d{2}-\d{2}t\d{2}:\d{2}:\d{2}", folded):
        return True
    if re.fullmatch(r"(nam\s+)?(?:19|20)\d{2}", folded_words):
        return True
    if any(term in folded_words for term in BOILERPLATE_TERMS):
        return True
    words = line.split()
    if len(words) <= 3 and not re.search(r"\d{4}|VNU|UET|USSH|ULIS|HSA|QHI|QHX|QHF", line):
        return True
    nav_terms = [
        "diem chuan",
        "de an tuyen sinh",
        "cac nganh dao tao",
        "to hop mon",
        "tu van chon truong",
        "ma truong",
        "ma nganh",
        "danh sach trung tuyen",
    ]
    if sum(1 for term in nav_terms if term in folded) >= 3:
        return True
    return False


def clean_document_text_for_corpus(text: str, title: str | None = None) -> str:
    lines = []
    seen_short_lines = set()
    in_manual_facts = False
    for raw_line in text.splitlines():
        line = normalize_text(raw_line)
        if not line:
            continue
        if line.startswith("Fact "):
            lines.append(line)
            continue
        if line.startswith("Curated factual QA annotations"):
            in_manual_facts = True
            continue
        if in_manual_facts:
            continue
        if is_boilerplate_line(line, title=title):
            continue
        key = line_key(line)
        if len(line.split()) < 18:
            if key in seen_short_lines:
                continue
            seen_short_lines.add(key)
        lines.append(line)
    return normalize_text("\n\n".join(lines))


def read_csv_rows(path: Path, required_fields: list[str]) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != required_fields:
            raise ValueError(f"{path} must have columns {required_fields}; got {reader.fieldnames}")
        rows = []
        for line_no, row in enumerate(reader, start=2):
            if row.get(None):
                raise ValueError(f"{path}:{line_no} has extra CSV columns: {row[None]}")
            cleaned = {field: clean_field(row[field]) for field in required_fields}
            missing = [field for field in required_fields if not cleaned[field]]
            if missing:
                raise ValueError(f"{path}:{line_no} has empty fields: {missing}")
            rows.append(cleaned)
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_title_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/").split("/")[-1] or parsed.netloc
    path = re.sub(r"[-_]+", " ", path)
    path = re.sub(r"\.(html?|pdf)$", "", path, flags=re.IGNORECASE)
    return path[:120] or parsed.netloc


def fetch_url(url: str, timeout: int) -> tuple[bytes, str]:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; VNU-RAG-Assignment/1.0)",
            "Accept": "text/html,application/pdf,*/*",
            "Accept-Encoding": "identity",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("content-type", "")
        content_encoding = response.headers.get("content-encoding", "").lower()
        body = response.read()
        if content_encoding == "gzip" or body[:2] == b"\x1f\x8b":
            body = gzip.decompress(body)
        elif content_encoding == "deflate":
            body = zlib.decompress(body)
        return body, content_type


def extract_html_text(body: bytes) -> tuple[str, str]:
    decoded = body.decode("utf-8", errors="replace")
    parser = VisibleTextParser()
    parser.feed(decoded)
    return parser.title, parser.text


def extract_pdf_text(body: bytes) -> str:
    try:
        from pypdf import PdfReader  # type: ignore

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(body)
            tmp_path = Path(tmp.name)
        try:
            reader = PdfReader(str(tmp_path))
            pages = [page.extract_text() or "" for page in reader.pages]
            return normalize_text("\n\n".join(pages))
        finally:
            tmp_path.unlink(missing_ok=True)
    except Exception:
        pass

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as pdf_file:
        pdf_file.write(body)
        pdf_path = Path(pdf_file.name)
    txt_path = pdf_path.with_suffix(".txt")
    try:
        subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), str(txt_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return normalize_text(txt_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return ""
    finally:
        pdf_path.unlink(missing_ok=True)
        txt_path.unlink(missing_ok=True)


def load_source_text(url: str, timeout: int, fetch_sources: bool) -> SourceText:
    if not fetch_sources:
        return SourceText(
            url=url,
            title=safe_title_from_url(url),
            text="",
            source_type="manual_qa_facts",
            fetch_status="skipped",
        )

    try:
        body, content_type = fetch_url(url, timeout)
        is_pdf = "pdf" in content_type.lower() or urlparse(url).path.lower().endswith(".pdf")
        if is_pdf:
            text = extract_pdf_text(body)
            title = safe_title_from_url(url)
            source_type = "pdf_source_text"
        else:
            title, text = extract_html_text(body)
            title = title or safe_title_from_url(url)
            source_type = "web_source_text"
        if text.count("\ufffd") > max(20, len(text) * 0.02):
            return SourceText(
                url=url,
                title=title,
                text=text,
                source_type="manual_qa_facts",
                fetch_status="decode_error",
                error="Fetched text looked like compressed or incorrectly decoded content.",
            )
        if len(text.split()) < 40:
            return SourceText(
                url=url,
                title=title,
                text=text,
                source_type="manual_qa_facts",
                fetch_status="too_short",
                error="Fetched text was too short; using manual facts as the reliable document body.",
            )
        return SourceText(url=url, title=title, text=text, source_type=source_type, fetch_status="ok")
    except HTTPError as exc:
        return SourceText(
            url=url,
            title=safe_title_from_url(url),
            text="",
            source_type="manual_qa_facts",
            fetch_status=f"http_{exc.code}",
            error=str(exc),
        )
    except URLError as exc:
        return SourceText(
            url=url,
            title=safe_title_from_url(url),
            text="",
            source_type="manual_qa_facts",
            fetch_status="url_error",
            error=str(exc),
        )
    except Exception as exc:
        return SourceText(
            url=url,
            title=safe_title_from_url(url),
            text="",
            source_type="manual_qa_facts",
            fetch_status="error",
            error=str(exc),
        )


def qa_fact_lines(rows: list[dict]) -> list[str]:
    lines = ["Curated factual QA annotations from the manual training dataset:"]
    for index, row in enumerate(rows, start=1):
        lines.append(f"Fact {index}. Question: {row['question']} Answer: {row['answer']}.")
    return lines


def classify_category(url: str, text: str) -> str:
    folded = ascii_fold(f"{url} {text[:4000]}")
    lowered = re.sub(r"[^a-z0-9]+", " ", folded)
    if any(term in lowered for term in ("tuition", "hoc phi", "fee")):
        return "tuition"
    if any(term in lowered for term in ("admission", "tuyen sinh", "diem chuan", "quota", "xet tuyen")):
        return "admission"
    if any(term in lowered for term in ("quy che", "regulation", "credits", "semester")):
        return "training_regulation"
    if any(term in lowered for term in ("history", "lich su", "established", "founded", "gioi thieu")):
        return "history"
    if any(term in lowered for term in ("program", "major", "dao tao", "curriculum", "nganh hoc")):
        return "academic_program"
    return "general"


def detect_source_language(text: str) -> str:
    sample = text[:4000]
    folded = ascii_fold(sample)
    accent_changes = sum(
        1
        for char in sample
        if char.isalpha() and ascii_fold(char) != char.lower()
    )
    vietnamese_terms = len(
        re.findall(
            r"\b(dai hoc|tuyen sinh|dao tao|hoc phi|quy che|diem chuan|sinh vien|chuong trinh)\b",
            folded,
        )
    )
    english_markers = len(re.findall(r"\b(the|and|of|university|program|admission|student)\b", folded))
    vietnamese_score = accent_changes + vietnamese_terms * 3
    if vietnamese_score > 8 and english_markers > 5:
        return "mixed"
    if vietnamese_score > 8:
        return "vi"
    return "en"


def build_documents(train_rows: list[dict], timeout: int, fetch_sources: bool) -> tuple[list[dict], dict[str, str]]:
    rows_by_url: dict[str, list[dict]] = {}
    for row in train_rows:
        rows_by_url.setdefault(row["source_url"], []).append(row)

    documents = []
    url_to_doc_id = {}
    collected_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for doc_index, url in enumerate(sorted(rows_by_url)):
        source = load_source_text(url, timeout=timeout, fetch_sources=fetch_sources)
        facts = "\n".join(qa_fact_lines(rows_by_url[url]))
        text_parts = []
        if source.text:
            text_parts.append(source.text)
        else:
            text_parts.append(facts)
        text = clean_document_text_for_corpus(normalize_text("\n\n".join(text_parts)), source.title)
        parsed = urlparse(url)
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        document_id = f"doc_{doc_index:04d}"
        document = {
            "id": document_id,
            "title": source.title,
            "url": url,
            "domain": parsed.netloc.lower(),
            "source_type": source.source_type,
            "source_language": detect_source_language(text),
            "qa_language": "en",
            "category": classify_category(url, text),
            "collected_at": collected_at,
            "fetch_status": source.fetch_status,
            "fetch_error": source.error,
            "manual_train_qa_count": len(rows_by_url[url]),
            "content_sha256": content_hash,
            "char_count": len(text),
            "word_count": len(text.split()),
            "text": text,
        }
        documents.append(document)
        url_to_doc_id[url] = document_id
    return documents, url_to_doc_id


def normalize_cached_documents(documents: list[dict], train_rows: list[dict] | None = None) -> list[dict]:
    rows_by_url: dict[str, list[dict]] = {}
    if train_rows:
        for row in train_rows:
            rows_by_url.setdefault(row["source_url"], []).append(row)

    cleaned_documents = []
    for document in documents:
        document = dict(document)
        source_lines = [
            line
            for line in document.get("text", "").splitlines()
            if not line.startswith("Fact ") and not line.startswith("Curated factual QA annotations")
        ]
        text_parts = ["\n".join(source_lines)]
        if document.get("url") in rows_by_url:
            document["manual_train_qa_count"] = len(rows_by_url[document["url"]])
        if not normalize_text(text_parts[0]) and document.get("url") in rows_by_url:
            text_parts.append("\n".join(qa_fact_lines(rows_by_url[document["url"]])))
        text = clean_document_text_for_corpus("\n\n".join(text_parts), document.get("title"))
        document["text"] = text
        document["source_language"] = detect_source_language(text)
        document["category"] = classify_category(document.get("url", ""), text)
        document["content_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        document["char_count"] = len(text)
        document["word_count"] = len(text.split())
        cleaned_documents.append(document)
    return cleaned_documents


def split_long_text(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int, str]]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [(0, len(words), " ".join(words))]

    pieces = []
    step = max(1, chunk_size - overlap)
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        pieces.append((start, end, " ".join(words[start:end])))
        if end >= len(words):
            break
        start += step
    return pieces


def is_useful_source_chunk(text: str) -> bool:
    words = text.split()
    if len(words) < 18:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    folded = ascii_fold(text)
    folded_words = re.sub(r"[^a-z0-9]+", " ", folded).strip()
    if any(term in folded_words for term in BOILERPLATE_TERMS):
        return False
    short_line_ratio = sum(1 for line in lines if len(line.split()) <= 5) / len(lines)
    long_line_count = sum(1 for line in lines if len(line.split()) >= 14)
    sentence_marks = len(re.findall(r"[.!?:;]", text))
    sentence_like_lines = sum(
        1 for line in lines if len(line.split()) >= 10 and re.search(r"[.!?:;]", line)
    )
    table_like_lines = sum(1 for line in lines if "|" in line)
    score_like_lines = sum(
        1 for line in lines if "|" in line and re.search(r"\b\d{2}\.\d{2}\b", line)
    )
    if table_like_lines >= 3 and score_like_lines >= 2:
        return True
    if len(lines) >= 8 and short_line_ratio >= 0.72 and long_line_count <= 1 and sentence_marks < 5:
        return False
    if len(lines) >= 10 and sentence_like_lines == 0 and long_line_count <= 1:
        return False
    useful_terms = [
        "dai hoc",
        "vnu",
        "uet",
        "ulis",
        "ussh",
        "tuyen sinh",
        "dao tao",
        "hoc phi",
        "quy che",
        "diem chuan",
        "chuong trinh",
        "nganh",
        "tin chi",
        "hsa",
        "ielts",
        "sat",
        "cong nghe",
        "ky thuat",
        "diem trung tuyen",
    ]
    if not any(term in folded_words for term in useful_terms):
        return False
    return True


def make_chunks(documents: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    chunks = []
    for document in documents:
        chunk_index = 0
        source_word_offset = 0
        pending_paragraphs: list[str] = []
        pending_start = 0

        def flush_pending() -> None:
            nonlocal chunk_index, pending_paragraphs, pending_start
            if not pending_paragraphs:
                return
            chunk_text = normalize_text("\n".join(pending_paragraphs))
            chunk_words = chunk_text.split()
            if chunk_words and is_useful_source_chunk(chunk_text):
                chunks.append(
                    {
                        "chunk_id": f"{document['id']}_chunk_{chunk_index:04d}",
                        "document_id": document["id"],
                        "chunk_index": chunk_index,
                        "title": document["title"],
                        "url": document["url"],
                        "category": document["category"],
                        "source_language": document["source_language"],
                        "qa_language": "en",
                        "start_word": pending_start,
                        "end_word": pending_start + len(chunk_words),
                        "word_count": len(chunk_words),
                        "chunk_source": "source_text",
                        "text": chunk_text,
                    }
                )
                chunk_index += 1
            pending_paragraphs = []

        for paragraph in [line.strip() for line in document["text"].splitlines() if line.strip()]:
            paragraph_words = paragraph.split()
            if not paragraph_words:
                continue
            paragraph_start = source_word_offset
            source_word_offset += len(paragraph_words)

            if paragraph.startswith("Fact "):
                flush_pending()
                chunks.append(
                    {
                        "chunk_id": f"{document['id']}_chunk_{chunk_index:04d}",
                        "document_id": document["id"],
                        "chunk_index": chunk_index,
                        "title": document["title"],
                        "url": document["url"],
                        "category": document["category"],
                        "source_language": document["source_language"],
                        "qa_language": "en",
                        "start_word": paragraph_start,
                        "end_word": paragraph_start + len(paragraph_words),
                        "word_count": len(paragraph_words),
                        "chunk_source": "manual_fact",
                        "text": paragraph,
                    }
                )
                chunk_index += 1
                continue

            if len(paragraph_words) > chunk_size:
                flush_pending()
                for local_start, local_end, chunk_text in split_long_text(paragraph, chunk_size, overlap):
                    chunk_words = chunk_text.split()
                    if not is_useful_source_chunk(chunk_text):
                        continue
                    chunks.append(
                        {
                            "chunk_id": f"{document['id']}_chunk_{chunk_index:04d}",
                            "document_id": document["id"],
                            "chunk_index": chunk_index,
                            "title": document["title"],
                            "url": document["url"],
                            "category": document["category"],
                            "source_language": document["source_language"],
                            "qa_language": "en",
                            "start_word": paragraph_start + local_start,
                            "end_word": paragraph_start + local_end,
                            "word_count": len(chunk_words),
                            "chunk_source": "source_text",
                            "text": chunk_text,
                        }
                    )
                    chunk_index += 1
                continue

            pending_words = sum(len(item.split()) for item in pending_paragraphs)
            if pending_paragraphs and pending_words + len(paragraph_words) > chunk_size:
                flush_pending()
            if not pending_paragraphs:
                pending_start = paragraph_start
            pending_paragraphs.append(paragraph)
        flush_pending()

        if chunk_index == 0 and document["text"].split():
            words = document["text"].split()
            chunk_text = " ".join(words)
            if is_useful_source_chunk(chunk_text):
                chunks.append(
                    {
                        "chunk_id": f"{document['id']}_chunk_{chunk_index:04d}",
                        "document_id": document["id"],
                        "chunk_index": chunk_index,
                        "title": document["title"],
                        "url": document["url"],
                        "category": document["category"],
                        "source_language": document["source_language"],
                        "qa_language": "en",
                        "start_word": 0,
                        "end_word": len(words),
                        "word_count": len(words),
                        "chunk_source": "source_text",
                        "text": chunk_text,
                    }
                )
    return chunks


def find_chunk_id(chunks_by_doc: dict[str, list[dict]], document_id: str, question: str, answer: str) -> str | None:
    doc_chunks = chunks_by_doc.get(document_id, [])
    answer_lower = answer.lower()
    question_lower = question.lower()
    for chunk in doc_chunks:
        text = chunk["text"].lower()
        if answer_lower in text and question_lower in text:
            return chunk["chunk_id"]
    for chunk in doc_chunks:
        if answer_lower in chunk["text"].lower():
            return chunk["chunk_id"]
    return doc_chunks[0]["chunk_id"] if doc_chunks else None


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_split_files(split: str, rows: list[dict]) -> None:
    folder = TRAIN_DIR if split == "train" else TEST_DIR
    folder.mkdir(parents=True, exist_ok=True)
    questions = "\n".join(row["question"] for row in rows)
    answers = "\n".join(row["answer"] for row in rows)
    (folder / "questions.txt").write_text(questions + ("\n" if rows else ""), encoding="utf-8")
    (folder / "reference_answers.txt").write_text(answers + ("\n" if rows else ""), encoding="utf-8")


def build_annotations(
    train_rows: list[dict],
    test_rows: list[dict],
    url_to_doc_id: dict[str, str],
    chunks: list[dict],
    seed: int,
) -> tuple[list[dict], list[dict]]:
    chunks_by_doc: dict[str, list[dict]] = {}
    for chunk in chunks:
        chunks_by_doc.setdefault(chunk["document_id"], []).append(chunk)

    rng = random.Random(seed)
    shuffled_train = list(train_rows)
    shuffled_test = list(test_rows)
    rng.shuffle(shuffled_train)
    rng.shuffle(shuffled_test)

    train_annotations = []
    for index, row in enumerate(shuffled_train):
        document_id = url_to_doc_id[row["source_url"]]
        chunk_id = find_chunk_id(chunks_by_doc, document_id, row["question"], row["answer"])
        train_annotations.append(
            {
                "id": f"train_{index:04d}",
                "split": "train",
                "question": row["question"],
                "answers": [row["answer"]],
                "answer": row["answer"],
                "source_id": document_id,
                "source_url": row["source_url"],
                "chunk_id": chunk_id,
                "qa_language": "en",
                "annotation_status": "manual_curated",
            }
        )

    test_annotations = []
    for index, row in enumerate(shuffled_test):
        test_annotations.append(
            {
                "id": f"test_{index:04d}",
                "split": "test",
                "question": row["question"],
                "answers": [row["answer"]],
                "answer": row["answer"],
                "qa_language": "en",
                "annotation_status": "manual_curated",
            }
        )

    return train_annotations, test_annotations


def write_readme(stats: dict) -> None:
    content = f"""# Manual English RAG Data for VNU QA

This dataset is built from the manually curated CSV files in
`data/manual_annotations/`. The CSV files are kept unchanged and can be reused
as the source of truth.

## Files

- `manual_annotations/train_qa.csv`: original manual training QA with source URLs.
- `manual_annotations/test_qa.csv`: original manual test QA without source URLs.
- `processed/documents.json`: document corpus grouped by train `source_url`.
- `processed/chunks.json`: retrieval chunks built from fetched source text. Manual
  QA facts are used only as a fallback when a source page has no usable text.
- `annotations/train_qa.json`: shuffled train annotations with source and chunk metadata.
- `annotations/test_qa.json`: shuffled test annotations.
- `train/questions.txt`, `train/reference_answers.txt`: train split in assignment format.
- `test/questions.txt`, `test/reference_answers.txt`: test split in assignment format.

## Counts

- Documents: {stats["document_count"]}
- Chunks: {stats["chunk_count"]}
- Train QA pairs: {stats["train_count"]}
- Test QA pairs: {stats["test_count"]}
- Unique train source URLs: {stats["unique_source_url_count"]}
- Shuffle seed: {stats["shuffle_seed"]}
- Chunk size/overlap: {stats["chunk_size"]}/{stats["chunk_overlap"]} words
- Source fetches with usable text: {stats["fetched_source_count"]}
- Source fetch fallbacks: {stats["fallback_source_count"]}

## Build Command

```bash
python scripts/build_manual_dataset.py --reuse-documents
```

If `processed/documents.json` is missing or you want to fetch source pages
again, run:

```bash
python scripts/build_manual_dataset.py --timeout 12
```
"""
    (DATA_DIR / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dataset outputs from manual QA CSV files.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic shuffle seed.")
    parser.add_argument("--chunk-size", type=int, default=140)
    parser.add_argument("--chunk-overlap", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--no-fetch-sources", action="store_true", help="Skip web/PDF fetching and build documents from manual facts only.")
    parser.add_argument("--reuse-documents", action="store_true", help="Reuse data/processed/documents.json and only rebuild chunks/splits.")
    args = parser.parse_args()

    before_hashes = {"train_csv": file_sha256(TRAIN_CSV), "test_csv": file_sha256(TEST_CSV)}
    train_rows = read_csv_rows(TRAIN_CSV, ["source_url", "question", "answer"])
    test_rows = read_csv_rows(TEST_CSV, ["question", "answer"])

    for folder in (PROCESSED_DIR, ANNOTATION_DIR, TRAIN_DIR, TEST_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    documents_path = PROCESSED_DIR / "documents.json"
    if args.reuse_documents:
        if not documents_path.exists():
            raise FileNotFoundError("Cannot use --reuse-documents because data/processed/documents.json is missing.")
        documents = normalize_cached_documents(json.loads(documents_path.read_text(encoding="utf-8")), train_rows)
        url_to_doc_id = {document["url"]: document["id"] for document in documents}
        missing_urls = sorted({row["source_url"] for row in train_rows} - set(url_to_doc_id))
        if missing_urls:
            raise ValueError(f"Cached documents are missing train source URLs: {missing_urls[:5]}")
    else:
        documents, url_to_doc_id = build_documents(
            train_rows,
            timeout=args.timeout,
            fetch_sources=not args.no_fetch_sources,
        )
    chunks = make_chunks(documents, chunk_size=args.chunk_size, overlap=args.chunk_overlap)
    train_annotations, test_annotations = build_annotations(
        train_rows,
        test_rows,
        url_to_doc_id,
        chunks,
        seed=args.seed,
    )

    write_split_files(
        "train",
        [{"question": row["question"], "answer": row["answer"]} for row in train_annotations],
    )
    write_split_files(
        "test",
        [{"question": row["question"], "answer": row["answer"]} for row in test_annotations],
    )
    write_json(PROCESSED_DIR / "documents.json", documents)
    write_json(PROCESSED_DIR / "chunks.json", chunks)
    write_json(ANNOTATION_DIR / "train_qa.json", train_annotations)
    write_json(ANNOTATION_DIR / "test_qa.json", test_annotations)

    fetched_source_count = sum(1 for doc in documents if doc["fetch_status"] == "ok")
    fallback_source_count = len(documents) - fetched_source_count
    stats = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_csv_hashes_before": before_hashes,
        "source_csv_hashes_after": {"train_csv": file_sha256(TRAIN_CSV), "test_csv": file_sha256(TEST_CSV)},
        "manual_csv_unchanged": before_hashes == {"train_csv": file_sha256(TRAIN_CSV), "test_csv": file_sha256(TEST_CSV)},
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "train_count": len(train_annotations),
        "test_count": len(test_annotations),
        "unique_source_url_count": len(url_to_doc_id),
        "shuffle_seed": args.seed,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "fetched_source_count": fetched_source_count,
        "fallback_source_count": fallback_source_count,
        "fetch_status_counts": {
            status: sum(1 for doc in documents if doc["fetch_status"] == status)
            for status in sorted({doc["fetch_status"] for doc in documents})
        },
    }
    write_json(PROCESSED_DIR / "dataset_metadata.json", stats)
    write_readme(stats)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    start = time.time()
    main()
