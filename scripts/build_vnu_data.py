"""Build English-first RAG data for Assignment 2.

The script crawls official VNU/UET pages, extracts HTML/PDF text, creates
documents/chunks, and writes at least 600 train QA pairs and 300 test QA pairs
in the required text-file format.

Generated QA pairs are extractive candidates with source/evidence metadata.
For the report, review the test split manually and compute IAA on a subset.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_HTML_DIR = RAW_DIR / "html"
RAW_PDF_DIR = RAW_DIR / "pdf"
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATION_DIR = DATA_DIR / "annotations"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"


USER_SEED_URLS = [
    "https://en.vnu.edu.vn/about-vnu/overview/history",
    "https://en.vnu.edu.vn/admission/why-vnu",
    "https://vnu.edu.vn/huong-dan-thuc-hien-cong-tac-tuyen-sinh-dai-hoc-chinh-quy-nam-2025-post36704.html",
    "https://en.vnu.edu.vn/about-vnu/organizational-structure/member-universities-schools",
    "https://uet.vnu.edu.vn/en/about-uet/",
    "https://uet.vnu.edu.vn/en/history-of-the-university/",
    "https://uet.vnu.edu.vn/en/academic-programs/",
    "https://uet.vnu.edu.vn/en/organizational-structure/",
    "https://uet.vnu.edu.vn/truong-dai-hoc-cong-nghe-dhqghn-ma-truong-qhi-thong-tin-tuyen-sinh-dai-hoc-nam-2025/",
    "https://uet.vnu.edu.vn/tuyen-sinh",
]


ADDITIONAL_SEED_URLS = [
    "https://en.vnu.edu.vn/about-vnu/overview/mission-vision-motto-strategy",
    "https://en.vnu.edu.vn/academic-programs/introduction",
    "https://en.vnu.edu.vn/academic-programs/undergraduate-programs",
    "https://en.vnu.edu.vn/academic-programs/graduate-and-research-programs",
    "https://en.vnu.edu.vn/academic-programs/international-programs",
    "https://en.vnu.edu.vn/about-vnu/overview/statistics",
    "https://en.vnu.edu.vn/about-vnu/overview/achievements",
    "https://en.vnu.edu.vn/about-vnu/organizational-structure/organization-chart",
    "https://en.vnu.edu.vn/about-vnu/organizational-structure/board-of-presidents",
    "https://uet.vnu.edu.vn/en/undergraduate-program/",
    "https://uet.vnu.edu.vn/en/educations-2/",
    "https://uet.vnu.edu.vn/en/educations/",
    "https://uet.vnu.edu.vn/en/computer-science-honors/",
    "https://uet.vnu.edu.vn/en/international-standard-program-computer-science/",
    "https://uet.vnu.edu.vn/en/bachelor-science-electronics-communication-engineering/",
    "https://uet.vnu.edu.vn/en/curriculum-bachelor-information-technology-high-quality-program/",
    "https://uet.vnu.edu.vn/en/advanced-institute-engineering-technology-avitech/",
    "https://www.uet.vnu.edu.vn/en/overview/",
    "https://uet.vnu.edu.vn/en/faculty-civil-engineering-vnu-university-engineering-technology/",
    "https://uet.vnu.edu.vn/wp-content/uploads/2023/05/Acceptable-research-fields-2024.pdf",
    "https://uet.vnu.edu.vn/wp-content/uploads/2017/02/Call-for-applications_CDE.pdf",
]


SEED_URLS = USER_SEED_URLS + [url for url in ADDITIONAL_SEED_URLS if url not in USER_SEED_URLS]


ALLOWED_DOMAINS = {
    "en.vnu.edu.vn",
    "vnu.edu.vn",
    "www.vnu.edu.vn",
    "files.vnu.edu.vn",
    "cdnportal.vnu.edu.vn",
    "uet.vnu.edu.vn",
    "www.uet.vnu.edu.vn",
    "fit.uet.vnu.edu.vn",
}


RELEVANT_URL_TERMS = [
    "about",
    "overview",
    "history",
    "mission",
    "vision",
    "statistics",
    "achievement",
    "organization",
    "organizational",
    "structure",
    "member",
    "school",
    "university",
    "academic",
    "program",
    "education",
    "undergraduate",
    "graduate",
    "master",
    "phd",
    "doctor",
    "international",
    "honors",
    "curriculum",
    "bachelor",
    "science",
    "engineering",
    "technology",
    "computer",
    "information",
    "electronics",
    "communication",
    "admission",
    "recruit",
    "enroll",
    "tuyen-sinh",
    "dao-tao",
    "quy-che",
    "chuong-trinh",
    "thong-tin",
    "diem-chuan",
]


RELEVANT_TEXT_TERMS = [
    "vietnam national university",
    "vnu",
    "vnu-uet",
    "university of engineering and technology",
    "academic program",
    "undergraduate",
    "graduate",
    "master",
    "ph.d",
    "admission",
    "curriculum",
    "training",
    "faculty",
    "research",
    "regulation",
    "credits",
    "program",
    "tuyen sinh",
    "dai hoc",
    "dhqghn",
    "truong dai hoc cong nghe",
]


BOILERPLATE_LINES = {
    "home",
    "about vnu",
    "overview",
    "academic programs",
    "admission",
    "contact",
    "sitemap",
    "search",
    "share",
    "you may be interested in",
    "related posts",
    "read more",
    "view more",
    "digital library",
    "library",
    "email",
    "map",
    "english",
    "vietnamese",
    "en",
    "vi",
}


END_MARKERS = [
    "You may be interested in",
    "Related posts",
    "Send a comment",
    "Share",
    "CONTACT US",
    "FANPAGE",
    "Copyright",
]


@dataclass
class FetchedResource:
    url: str
    status: str
    content_type: str
    body: bytes
    error: str | None = None


class VisibleTextParser(HTMLParser):
    SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas"}
    BREAK_TAGS = {
        "article",
        "br",
        "caption",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "main",
        "p",
        "section",
        "td",
        "th",
        "tr",
        "ul",
        "ol",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.links: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_dict = dict(attrs)
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
        if tag == "a" and attrs_dict.get("href"):
            self.links.append(attrs_dict["href"] or "")
        if self.skip_depth == 0 and tag in self.BREAK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
        if self.skip_depth == 0 and tag in self.BREAK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth == 0 and data:
            self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


def ensure_dirs(clean: bool) -> None:
    for path in [RAW_HTML_DIR, RAW_PDF_DIR, PROCESSED_DIR, ANNOTATION_DIR, TRAIN_DIR, TEST_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    if clean:
        for folder in [RAW_HTML_DIR, RAW_PDF_DIR, PROCESSED_DIR, ANNOTATION_DIR, TRAIN_DIR, TEST_DIR]:
            for child in folder.iterdir():
                if child.is_file():
                    child.unlink()


def safe_filename(url: str, suffix: str) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", urlparse(url).path.strip("/").lower()).strip("-")
    slug = slug[:90] or "index"
    return f"{slug}-{digest}{suffix}"


def normalize_url(base_url: str, link: str) -> str | None:
    if not link:
        return None
    if link.startswith(("mailto:", "tel:", "javascript:", "#")):
        return None
    absolute = urljoin(base_url, link)
    absolute, _fragment = urldefrag(absolute)
    parsed = urlparse(absolute)
    if parsed.scheme not in {"http", "https"}:
        return None
    if parsed.netloc.lower() not in ALLOWED_DOMAINS:
        return None
    lowered_path = parsed.path.lower()
    blocked_ext = (
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".svg",
        ".zip",
        ".rar",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
    )
    if lowered_path.endswith(blocked_ext):
        return None
    return absolute


def is_relevant_url(url: str) -> bool:
    lowered = url.lower()
    if lowered.endswith(".pdf"):
        return True
    return any(term in lowered for term in RELEVANT_URL_TERMS)


def fetch_url(url: str, timeout: int, retries: int = 1) -> FetchedResource:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; VNU-RAG-Assignment-DataBuilder/3.0; "
            "+https://en.vnu.edu.vn)"
        )
    }
    for attempt in range(retries + 1):
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=timeout) as response:
                content_type = response.headers.get("Content-Type", "")
                body = response.read()
            return FetchedResource(url=url, status="ok", content_type=content_type, body=body)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            if attempt >= retries:
                return FetchedResource(url=url, status="error", content_type="", body=b"", error=str(exc))
            time.sleep(0.4)
    return FetchedResource(url=url, status="error", content_type="", body=b"", error="unknown error")


def decode_response_text(resource: FetchedResource) -> str:
    content_type = resource.content_type or ""
    charset_match = re.search(r"charset=([\w.-]+)", content_type, flags=re.I)
    charset = charset_match.group(1) if charset_match else "utf-8"
    return resource.body.decode(charset, errors="replace")


def decode_js_escapes(value: str) -> str:
    def replace_unicode(match: re.Match[str]) -> str:
        return chr(int(match.group(1), 16))

    value = re.sub(r"\\u([0-9a-fA-F]{4})", replace_unicode, value)
    value = value.replace('\\"', '"')
    value = value.replace("\\/", "/")
    value = value.replace("\\n", "\n")
    value = value.replace("\\t", " ")
    return html.unescape(value)


def parse_visible_text(html_text: str) -> tuple[str, list[str]]:
    parser = VisibleTextParser()
    parser.feed(html_text)
    parser.close()
    return parser.text(), parser.links


def extract_html_text_and_links(html_text: str, base_url: str) -> tuple[str, list[str]]:
    visible_text, raw_links = parse_visible_text(html_text)

    embedded_texts = []
    for body in re.findall(r"<script[^>]*>(.*?)</script>", html_text, flags=re.I | re.S):
        if "\\u003c" not in body and "dangerouslySetInnerHTML" not in body:
            continue
        decoded = decode_js_escapes(body)
        decoded_text, _ = parse_visible_text(decoded)
        embedded_texts.append(decoded_text)

    combined = "\n".join([visible_text, *embedded_texts])
    links = []
    for link in raw_links:
        normalized = normalize_url(base_url, link)
        if normalized:
            links.append(normalized)
    return clean_text(combined), sorted(set(links))


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    lines = []
    seen = set()
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip(" -|\t")
        if not line:
            continue
        lowered = line.lower()
        if lowered in BOILERPLATE_LINES:
            continue
        if len(line) <= 16 and not re.search(r"\d", line):
            continue
        if line.startswith(("http://", "https://")) and len(line) < 100:
            continue
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)

    text = "\n".join(lines)
    for marker in END_MARKERS:
        index = text.lower().find(marker.lower(), 1200)
        if index > 0:
            text = text[:index]
    return text.strip()


def extract_title(html_text: str, fallback: str) -> str:
    patterns = [
        r'<meta\s+property=["\']og:title["\']\s+content=["\'](.*?)["\']',
        r"<title[^>]*>(.*?)</title>",
        r"<h1[^>]*>(.*?)</h1>",
    ]
    for pattern in patterns:
        match = re.search(pattern, html_text, flags=re.I | re.S)
        if not match:
            continue
        text, _ = parse_visible_text(match.group(1))
        title = clean_text(text or match.group(1)).replace("\n", " ").strip()
        if title:
            return title[:180]
    return fallback


def extract_pdf_text(pdf_path: Path) -> str:
    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        return ""
    try:
        result = subprocess.run(
            [pdftotext, "-layout", str(pdf_path), "-"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=90,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return clean_text(result.stdout)


def detect_language(text: str) -> str:
    lowered = text[:4000].lower()
    vietnamese_hits = sum(lowered.count(term) for term in [" dhqghn", "tuyen sinh", "dai hoc", "nganh", "diem chuan"])
    english_hits = sum(lowered.count(term) for term in ["vietnam national university", "university", "program", "admission"])
    return "vi" if vietnamese_hits > english_hits + 2 else "en"


def classify_category(url: str, title: str, text: str) -> str:
    sample = " ".join([url, title, text[:1200]]).lower()
    if "admission" in sample or "tuyen-sinh" in sample or "enroll" in sample:
        return "admission"
    if "curriculum" in sample or "academic program" in sample or "undergraduate" in sample or "graduate" in sample:
        return "academic_program"
    if "organization" in sample or "structure" in sample or "member" in sample:
        return "organization"
    if "history" in sample or "mission" in sample or "vision" in sample or "overview" in sample:
        return "overview_history"
    if "research" in sample or "institute" in sample or "faculty" in sample:
        return "research_faculty"
    return "general"


def text_is_relevant(text: str, url: str) -> bool:
    lowered = (" ".join([url, text[:5000]])).lower()
    return any(term in lowered for term in RELEVANT_TEXT_TERMS)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def crawl_sources(args: argparse.Namespace, collected_at: str) -> tuple[list[dict], list[dict]]:
    queue = deque(SEED_URLS)
    seen_urls = set()
    seen_hashes = set()
    documents: list[dict] = []
    failures: list[dict] = []
    discovered_urls: list[str] = []
    mandatory_urls = set(USER_SEED_URLS)

    while queue and len(seen_urls) < args.max_fetches and len(documents) < args.max_documents:
        url = queue.popleft()
        if url in seen_urls:
            continue
        seen_urls.add(url)
        if url not in mandatory_urls and not is_relevant_url(url):
            continue

        resource = fetch_url(url, timeout=args.timeout)
        if resource.status != "ok":
            failures.append({"url": url, "error": resource.error})
            continue

        parsed = urlparse(url)
        is_pdf = parsed.path.lower().endswith(".pdf") or "pdf" in (resource.content_type or "").lower()
        links: list[str] = []

        if is_pdf:
            raw_path = RAW_PDF_DIR / safe_filename(url, ".pdf")
            raw_path.write_bytes(resource.body)
            text = extract_pdf_text(raw_path)
            title = Path(parsed.path).name or "PDF document"
            source_type = "pdf"
        else:
            html_text = decode_response_text(resource)
            raw_path = RAW_HTML_DIR / safe_filename(url, ".html")
            raw_path.write_text(html_text, encoding="utf-8")
            text, links = extract_html_text_and_links(html_text, url)
            title = extract_title(html_text, fallback=url)
            source_type = "html"

        for link in links:
            if link not in seen_urls and (is_relevant_url(link) or "/en/" in link.lower() or link in mandatory_urls):
                queue.append(link)
                discovered_urls.append(link)

        if len(text.split()) < args.min_words:
            continue
        if url not in mandatory_urls and not text_is_relevant(text, url):
            continue

        content_hash = sha256_text(text)
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)

        language = detect_language(text)
        document = {
            "id": f"doc_{len(documents):04d}",
            "title": title,
            "url": url,
            "domain": parsed.netloc.lower(),
            "source_type": source_type,
            "source_language": language,
            "qa_language": "en",
            "category": classify_category(url, title, text),
            "collected_at": collected_at,
            "raw_path": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
            "content_sha256": content_hash,
            "char_count": len(text),
            "word_count": len(text.split()),
            "text": text,
        }
        documents.append(document)

        if len(documents) % 25 == 0:
            print(f"Collected {len(documents)} documents; queue={len(queue)}", flush=True)
        if args.delay:
            time.sleep(args.delay)

    write_json(
        RAW_DIR / "source_urls.json",
        {
            "user_seed_urls": USER_SEED_URLS,
            "additional_seed_urls": ADDITIONAL_SEED_URLS,
            "allowed_domains": sorted(ALLOWED_DOMAINS),
            "fetched_url_count": len(seen_urls),
            "discovered_url_count": len(set(discovered_urls)),
            "document_url_count": len(documents),
            "discovered_urls_sample": sorted(set(discovered_urls))[:800],
        },
    )
    return documents, failures


def make_chunks(documents: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    chunks: list[dict] = []
    step = max(1, chunk_size - overlap)
    for document in documents:
        words = document["text"].split()
        start = 0
        index = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunks.append(
                {
                    "chunk_id": f"{document['id']}_chunk_{index:04d}",
                    "document_id": document["id"],
                    "chunk_index": index,
                    "title": document["title"],
                    "url": document["url"],
                    "category": document["category"],
                    "source_language": document["source_language"],
                    "qa_language": "en",
                    "start_word": start,
                    "end_word": end,
                    "word_count": len(chunk_words),
                    "text": " ".join(chunk_words),
                }
            )
            if end >= len(words):
                break
            start += step
            index += 1
    return chunks


def auto_chunk(documents: list[dict], preferred_size: int, preferred_overlap: int) -> tuple[int, int, list[dict]]:
    candidates = [
        (preferred_size, preferred_overlap),
        (90, 25),
        (75, 20),
        (60, 15),
        (50, 12),
    ]
    best = (preferred_size, preferred_overlap, make_chunks(documents, preferred_size, preferred_overlap))
    for size, overlap in candidates:
        chunks = make_chunks(documents, size, overlap)
        best = (size, overlap, chunks)
        if len(chunks) >= 1200:
            break
    return best


def sentence_split(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text)
    pieces = re.split(r"(?<=[.!?])\s+|(?<=:)\s+|(?<=;)\s+", normalized)
    sentences = []
    for piece in pieces:
        sentence = piece.strip(" -")
        if 45 <= len(sentence) <= 420 and is_good_sentence(sentence):
            sentences.append(sentence)
    return sentences


def normalize_sentence_for_qa(sentence: str) -> str:
    sentence = re.sub(r"\s+", " ", sentence).strip()
    nav_terms = ["VNU Journal of Science", "Organizational structure", "Research Centers", "Covid-19", "Digital Library"]
    cut_patterns = [
        r"\bVNU-UET\s+presently\s+offers\b",
        r"\bThe VNU University of Engineering and Technology\b",
        r"\bEvery year\b",
        r"\bIn\s+(?:19|20)\d{2},\b",
        r"\bVNU\s+(?:was|is|has|offers|produces|reports)\b",
        r"\bVietnam National University\b",
        r"\bThe Faculty\b",
        r"\bThe College of Technology\b",
    ]
    for pattern in cut_patterns:
        match = re.search(pattern, sentence)
        if not match or match.start() <= 20:
            continue
        prefix = sentence[: match.start()]
        if any(term in prefix for term in nav_terms) or len(prefix.split()) > 12:
            sentence = sentence[match.start() :]
            break
    return sentence.strip(" -")


def is_good_sentence(sentence: str) -> bool:
    words = sentence.split()
    if len(words) < 8:
        return False
    letters = sum(char.isalpha() for char in sentence)
    digits = sum(char.isdigit() for char in sentence)
    if letters / max(1, len(sentence)) < 0.52:
        return False
    if digits > letters * 0.45:
        return False
    lowered = sentence.lower()
    useful_terms = [
        " is ",
        " are ",
        " was ",
        " were ",
        " has ",
        " have ",
        " offers ",
        " provides ",
        " established ",
        " founded ",
        " ranked ",
        " includes ",
        " consists ",
        " aims ",
        " requires ",
        " admitted ",
        " admission ",
        " program",
        " university",
        " faculty",
        " school",
        " institute",
    ]
    return any(term in f" {lowered} " for term in useful_terms)


ANSWER_PATTERNS: list[tuple[str, str]] = [
    ("date", r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b"),
    ("date", r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}\b"),
    ("numeric_date", r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b"),
    ("year", r"\b(?:19|20)\d{2}\b"),
    ("rank", r"\b\d{1,4}(?:st|nd|rd|th)\b"),
    ("percent", r"\b\d{1,3}(?:\.\d+)?\s*%\b"),
    ("credits", r"\b\d{1,3}\s+credits?\b"),
    ("duration", r"\b\d+(?:\.\d+)?\s+years?\b"),
    ("count", r"\b\d{1,4}(?:,\d{3})?\s+(?:undergraduate|master|Ph\.D\.|doctoral|doctor|programs?|learners?|students?|bachelors?|staff|lecturers?|articles?|credits?|faculties|departments|institutes|schools|universities|centers?)\b"),
    ("decision", r"\b(?:No\.|Decision\s+No\.|Decision)\s*[A-Z0-9./-]+\b"),
    ("code", r"\b(?:QHI|VNU-UET|VNU|UET|AUN|ABET|CDIO|ACM|IEEE|IELTS|TOEFL|SAT|HSA|CN\d{1,2}|QH[A-Z])\b"),
    ("email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ("phone", r"\(?\+?\d{2,3}\)?[\s.-]?\d{2,4}[\s.-]?\d{3,4}[\s.-]?\d{3,4}\b"),
]


PROPER_NOUN_PATTERN = re.compile(
    r"\b(?:[A-Z][A-Za-z&.-]+|VNU|UET|Ph\.D\.|AUN|ABET|CDIO|ACM|IEEE)"
    r"(?:\s+(?:of|and|for|in|the|&|[A-Z][A-Za-z&.-]+|VNU|UET|Ph\.D\.|AUN|ABET|CDIO|ACM|IEEE)){1,8}"
)


def compact_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title).strip(" -|")
    title = re.sub(r"\s+-\s+VNU.*$", "", title)
    return title[:110] or "the source document"


def clean_subject(value: str, fallback: str) -> str:
    value = re.sub(r"\([^)]{0,80}\)", "", value)
    value = re.sub(r"^(?:In|On|By|As of|Since)\s+[^,]{1,80},\s*", "", value, flags=re.I)
    value = re.sub(r"^(?:Every year|Currently|Presently|Besides|Moreover|Therefore),?\s*", "", value, flags=re.I)
    value = re.sub(r"\s+", " ", value).strip(" ,.;:-")
    value = re.sub(r"^(?:the|a|an)\s+", "", value, flags=re.I)
    words = value.split()
    if len(words) > 13:
        value = " ".join(words[-13:])
    if len(value) < 3 or value.lower() in {"it", "they", "there", "this", "these", "those"}:
        return fallback
    return value


def sentence_subject(sentence: str, title: str) -> str:
    if re.search(r"\bVNU-UET\s+presently\s+offers\b", sentence):
        return "VNU-UET"
    if "the university was honored" in sentence.lower() and "VNU" in sentence:
        return "VNU"
    if re.search(r"\bVNU\s+(?:was|is|has|offers|produces|reports)\b", sentence):
        return "VNU"
    patterns = [
        r"^(?P<subject>.+?)\s+(?:presently\s+)?(?:offers|provides|has|have|includes|consists|produces|requires|aims|reports)\b",
        r"^(?P<subject>.+?)\s+(?:was|were|is|are)\s+(?:founded|established|reorganized|renamed|known|ranked|listed|located|approved|issued|honored)\b",
        r"^(?P<subject>.+?)\s+(?:founded|established|reorganized|renamed)\b",
        r"^(?:In|On)\s+[^,]{1,80},\s+(?P<subject>.+?)\s+(?:was|were|is|are|has|have|offers|provides|ranked|issued)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, sentence, flags=re.I)
        if match:
            return clean_subject(match.group("subject"), compact_title(title))
    return compact_title(title)


def answer_unit(answer: str, answer_type: str, sentence: str) -> str:
    if answer_type.startswith("count:"):
        return answer_type.split(":", 1)[1]
    if answer_type == "count":
        match = re.match(r"\d{1,4}(?:,\d{3})?\s+(.+)", answer)
        if match:
            unit = match.group(1).strip()
            if unit.isupper():
                return unit.lower()
            if unit in {"master", "doctor", "doctoral", "undergraduate"}:
                return f"{unit} programs"
            return unit
    if answer_type == "credits":
        return "credits"
    return "items"


def normalize_reference_answer(answer: str, answer_type: str) -> str:
    if answer_type == "count":
        match = re.match(r"(\d{1,4}(?:,\d{3})?)\s+", answer)
        if match:
            return match.group(1)
    if answer_type == "credits":
        match = re.match(r"(\d{1,3})\s+credits?", answer, flags=re.I)
        if match:
            return match.group(1)
    return answer


def detect_action(sentence: str) -> str:
    actions = [
        "founded",
        "established",
        "reorganized",
        "renamed",
        "issued",
        "approved",
        "honored",
        "ranked",
        "launched",
        "created",
        "formed",
    ]
    lowered = sentence.lower()
    for action in actions:
        if action in lowered:
            return action
    return "mentioned"


def code_question(sentence: str, answer: str, title: str) -> str:
    before = sentence.split(answer, 1)[0]
    paren_match = re.search(r"([A-Z][A-Za-z0-9&.,'\-\s]{4,90})\s*\($", before)
    if paren_match:
        entity = clean_subject(paren_match.group(1), compact_title(title))
        if len(entity.split()) < 2 and entity not in {"VNU", "UET"}:
            return f"Which abbreviation is used in {compact_title(title)}?"
        return f"What abbreviation is used for {entity}?"
    if re.search(r"school code|code|mã trường", sentence, flags=re.I):
        return f"What school code is given for {compact_title(title)}?"
    if answer in {"IELTS", "TOEFL", "SAT", "HSA"}:
        return f"Which test or certificate is mentioned in {compact_title(title)}?"
    return f"Which abbreviation is used in {compact_title(title)}?"


def entity_question(sentence: str, answer: str, title: str) -> str:
    subject = sentence_subject(sentence, title)
    entity_type = "entity"
    if "University" in answer:
        entity_type = "university"
    elif "School" in answer:
        entity_type = "school"
    elif "Faculty" in answer:
        entity_type = "faculty"
    elif "Institute" in answer:
        entity_type = "institute"
    elif "Center" in answer or "Centre" in answer:
        entity_type = "center"
    elif "Program" in answer or "program" in answer:
        entity_type = "program"
    elif any(word in answer for word in ["Ministry", "Department", "Council", "Committee"]):
        entity_type = "organization"
    elif any(word in answer for word in ["Science", "Studies", "Engineering", "Technology", "Management"]):
        entity_type = "field"

    if re.search(r"\b(?:Prof\.|Professor|Dr\.|Assoc\.|Nguyen|Tran|Le|Pham)\b", answer):
        if "led by" in sentence.lower():
            return f"Who led {subject}?"
        if "director" in sentence.lower():
            return f"Who is listed as director in {compact_title(title)}?"
        return f"Who is mentioned in relation to {subject}?"
    if "known" in sentence.lower() or "called" in sentence.lower():
        return f"What name is {subject} known by?"
    if answer in sentence[: len(answer) + 5]:
        after = sentence[len(answer) :].strip(" ,.;:-")
        after = re.sub(r"\([^)]{0,80}\)", "", after)
        after = re.sub(r"\s+", " ", after).strip()
        if 12 <= len(after) <= 190 and not after.startswith(("and ", "or ", ",")):
            after = after[0].lower() + after[1:]
            return f"Which {entity_type} {after.rstrip('.')}?"
    return f"Which {entity_type} is mentioned in {compact_title(title)}?"


def title_from_url(url: str) -> str:
    path = urlparse(url).path.strip("/")
    slug = path.split("/")[-1] or path.split("/")[-2] if "/" in path else path
    slug = re.sub(r"-post\d+.*$", "", slug)
    slug = re.sub(r"\.(html?|php|pdf)$", "", slug)
    words = [part for part in re.split(r"[-_]+", slug) if part]
    if not words:
        return "Source document"
    title = " ".join(words).title()
    for old, new in {
        "Vnu": "VNU",
        "Uet": "UET",
        "Dhqghn": "DHQGHN",
        "Phd": "PhD",
        "Pdf": "PDF",
    }.items():
        title = title.replace(old, new)
    return title[:140]


def repair_document_metadata(documents: list[dict]) -> list[dict]:
    for document in documents:
        title = str(document.get("title", "")).strip()
        bad_title = title.startswith("http") or not title or "Digital LibraryLibrary" in title
        if bad_title:
            document["title"] = title_from_url(document.get("url", ""))
        document["category"] = classify_category(document.get("url", ""), document.get("title", ""), document.get("text", ""))
    return documents


def make_question(document: dict, sentence: str, answer: str, answer_type: str, index: int) -> str:
    title = compact_title(document["title"])
    subject = sentence_subject(sentence, title)
    action = detect_action(sentence)

    if answer_type in {"date", "numeric_date"}:
        if action != "mentioned":
            return f"When was {subject} {action}?"
        return f"What date is given for {subject}?"
    if answer_type == "year":
        if action != "mentioned":
            return f"In what year was {subject} {action}?"
        if sentence.lower().startswith("in "):
            return f"What year is mentioned in {title}?"
        return f"What year is given for {subject}?"
    if answer_type == "rank":
        return f"What rank did {subject} achieve?"
    if answer_type == "percent":
        return f"What percentage is reported for {subject}?"
    if answer_type == "credits":
        return f"How many credits are mentioned for {subject}?"
    if answer_type == "duration":
        return f"What duration is stated for {subject}?"
    if answer_type.startswith("count"):
        unit = answer_unit(answer, answer_type, sentence)
        if "offers" in sentence.lower():
            return f"How many {unit} does {subject} offer?"
        if "produces" in sentence.lower():
            return f"How many {unit} does {subject} produce?"
        if "has" in sentence.lower() or "have" in sentence.lower():
            return f"How many {unit} does {subject} have?"
        return f"How many {unit} are mentioned for {subject}?"
    if answer_type == "decision":
        return f"What decision or document number is given for {subject}?"
    if answer_type == "code":
        return code_question(sentence, answer, title)
    if answer_type == "email":
        return f"What email address is listed for {subject}?"
    if answer_type == "phone":
        return f"What phone number is listed for {subject}?"
    return entity_question(sentence, answer, title)


def find_chunk_id(chunks_by_doc: dict[str, list[dict]], document_id: str, sentence: str) -> str | None:
    for chunk in chunks_by_doc.get(document_id, []):
        if sentence[:80] in chunk["text"]:
            return chunk["chunk_id"]
    sentence_words = set(sentence.lower().split()[:20])
    best_id = None
    best_score = 0
    for chunk in chunks_by_doc.get(document_id, []):
        score = len(sentence_words.intersection(chunk["text"].lower().split()))
        if score > best_score:
            best_score = score
            best_id = chunk["chunk_id"]
    return best_id


def answer_candidates(sentence: str) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for answer_type, pattern in ANSWER_PATTERNS:
        for match in re.finditer(pattern, sentence, flags=re.I):
            raw_answer = match.group(0).strip(" ,.;:")
            if answer_type == "code":
                lowered_sentence = sentence.lower()
                if raw_answer.lower() in {"vnu", "uet"} and "code" not in lowered_sentence and "(" not in sentence:
                    continue
            if answer_type == "count":
                unit = answer_unit(raw_answer, answer_type, sentence)
                if unit.isupper():
                    continue
                answer = normalize_reference_answer(raw_answer, answer_type)
                candidates.append((answer, f"count:{unit}"))
            else:
                candidates.append((normalize_reference_answer(raw_answer, answer_type), answer_type))
    for match in PROPER_NOUN_PATTERN.finditer(sentence):
        answer = match.group(0).strip(" ,.;:")
        if 4 <= len(answer) <= 90 and len(answer.split()) <= 10:
            candidates.append((answer, "entity"))
    deduped = []
    seen = set()
    for answer, answer_type in candidates:
        key = answer.lower()
        if key in seen:
            continue
        if answer.lower() in {"the university", "the program", "the source", "faculty", "school"}:
            continue
        if answer_type == "entity" and len(answer.split()) == 1 and answer.isalpha() and answer not in {"VNU", "UET"}:
            continue
        if answer_type == "entity" and re.search(r"\b(?:of|and|for|in|the|from|with|to)$", answer, flags=re.I):
            continue
        if answer_type == "entity" and len(answer.split()) >= 2 and len(answer.split()[-1]) <= 2:
            continue
        if "Digital Library" in answer:
            continue
        seen.add(key)
        deduped.append((answer, answer_type))
    return deduped


def generate_qa(documents: list[dict], chunks: list[dict], target_total: int) -> list[dict]:
    chunks_by_doc: dict[str, list[dict]] = {}
    for chunk in chunks:
        chunks_by_doc.setdefault(chunk["document_id"], []).append(chunk)

    preferred_categories = {
        "admission": 0,
        "academic_program": 1,
        "overview_history": 2,
        "organization": 3,
        "research_faculty": 4,
        "general": 5,
    }
    sorted_documents = sorted(documents, key=lambda doc: (preferred_categories.get(doc["category"], 9), doc["id"]))

    qa_items: list[dict] = []
    seen = set()
    pass_limits = [2, 5, 99]
    for pass_limit in pass_limits:
        for document in sorted_documents:
            per_doc = 0
            for sentence in sentence_split(document["text"]):
                sentence = normalize_sentence_for_qa(sentence)
                for answer, answer_type in answer_candidates(sentence):
                    question = make_question(document, sentence, answer, answer_type, len(qa_items))
                    key = (question.lower(), answer.lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    qa_items.append(
                        {
                            "id": f"qa_{len(qa_items):05d}",
                            "question": question,
                            "answers": [answer],
                            "source_id": document["id"],
                            "source_title": document["title"],
                            "source_url": document["url"],
                            "chunk_id": find_chunk_id(chunks_by_doc, document["id"], sentence),
                            "question_type": answer_type,
                            "evidence": sentence,
                            "source_language": document["source_language"],
                            "qa_language": "en",
                            "annotation_status": "auto_candidate_needs_human_review",
                        }
                    )
                    per_doc += 1
                    if len(qa_items) >= target_total:
                        return qa_items
                    if per_doc >= pass_limit:
                        break
                if per_doc >= pass_limit:
                    break
    return qa_items


def qa_quality_score(row: dict) -> int:
    question = row["question"]
    question_type = row["question_type"].split(":", 1)[0]
    score = 0
    if question.startswith(("When ", "In what year ", "How many ", "What rank ", "What percentage ", "What school code ", "Who ")):
        score += 20
    if question.startswith("What abbreviation "):
        score += 16
    if question.startswith(("Which university ", "Which school ", "Which faculty ", "Which institute ", "Which center ", "Which program ", "Which organization ", "Which field ")):
        score += 10
    if question_type != "entity":
        score += 12
    if row.get("source_language") == "en":
        score += 4
    if " is discussed in " in question:
        score -= 8
    if question.startswith("Which acronym appears"):
        score -= 12
    if question.count("?") == 1:
        score += 2
    if len(question) > 180:
        score -= 4
    return score


def write_qa_splits(qa_items: list[dict], train_target: int, test_target: int) -> dict[str, int]:
    ranked = sorted(qa_items, key=qa_quality_score, reverse=True)
    selected = ranked[: train_target + test_target]
    test = []
    train = []
    for index, row in enumerate(selected):
        if len(test) < test_target and index % 3 == 2:
            test.append(row)
        elif len(train) < train_target:
            train.append(row)
        elif len(test) < test_target:
            test.append(row)
    if len(test) < test_target:
        for row in selected:
            if row not in train and row not in test:
                test.append(row)
                if len(test) >= test_target:
                    break
    for split, rows, folder in [("train", train, TRAIN_DIR), ("test", test, TEST_DIR)]:
        for index, row in enumerate(rows):
            row["split"] = split
            row["id"] = f"{split}_{index:04d}"
        (folder / "questions.txt").write_text(
            "\n".join(row["question"] for row in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )
        (folder / "reference_answers.txt").write_text(
            "\n".join(";".join(row["answers"]) for row in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )
        write_json(ANNOTATION_DIR / f"{split}_qa.json", rows)
    return {"train": len(train), "test": len(test)}


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8").splitlines())


def write_readmes(stats: dict) -> None:
    data_readme = f"""# English RAG Data for VNU/UET QA

This dataset was built from public official VNU/UET sources. The main seed URLs
come from the assignment data owner, and the script also discovers additional
official pages and PDFs from the same domains.

## Files

- `raw/source_urls.json`: seed URLs, allowed domains, and discovered URL sample.
- `raw/html/`, `raw/pdf/`: downloaded raw source snapshots.
- `processed/documents.json`: cleaned document corpus.
- `processed/chunks.json`: retrieval chunks.
- `annotations/train_qa.json`, `annotations/test_qa.json`: QA pairs with source, chunk, and evidence.
- `train/questions.txt`, `train/reference_answers.txt`: train split in assignment format.
- `test/questions.txt`, `test/reference_answers.txt`: test split in assignment format.

## Counts

- Documents: {stats["document_count"]}
- Chunks: {stats["chunk_count"]}
- Train QA pairs: {stats["qa_counts"]["train"]}
- Test QA pairs: {stats["qa_counts"]["test"]}
- Chunk size/overlap: {stats["chunk_size_words"]}/{stats["chunk_overlap_words"]} words

## Quality Note

QA pairs are extractive candidates generated from source evidence and marked
`auto_candidate_needs_human_review`. Before submission, manually review the test
split and have at least two team members annotate a random subset to compute IAA.
"""
    (DATA_DIR / "README.md").write_text(data_readme, encoding="utf-8")

    root_readme = """# VNU/UET RAG Data Builder

Build the assignment data:

```bash
python scripts/build_vnu_data.py
```

The generated dataset is written under `data/`.
"""
    (ROOT / "README.md").write_text(root_readme, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--delay", type=float, default=0.02)
    parser.add_argument("--max-fetches", type=int, default=650)
    parser.add_argument("--max-documents", type=int, default=260)
    parser.add_argument("--min-words", type=int, default=70)
    parser.add_argument("--chunk-size", type=int, default=110)
    parser.add_argument("--chunk-overlap", type=int, default=25)
    parser.add_argument("--target-train", type=int, default=600)
    parser.add_argument("--target-test", type=int, default=300)
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--from-cache", action="store_true", help="Reuse data/processed/documents.json instead of crawling.")
    args = parser.parse_args()

    if args.chunk_overlap >= args.chunk_size:
        parser.error("--chunk-overlap must be smaller than --chunk-size")

    ensure_dirs(clean=(not args.no_clean and not args.from_cache))
    collected_at = datetime.now(timezone.utc).isoformat()

    if args.from_cache:
        documents_path = PROCESSED_DIR / "documents.json"
        if not documents_path.exists():
            print("Cannot use --from-cache because data/processed/documents.json is missing.", file=sys.stderr)
            return 2
        documents = repair_document_metadata(json.loads(documents_path.read_text(encoding="utf-8")))
        failures = []
        print(f"Loaded {len(documents)} cached documents.", flush=True)
    else:
        print("Crawling official VNU/UET sources...", flush=True)
        documents, failures = crawl_sources(args, collected_at)

    print("Building chunks...", flush=True)
    chunk_size, overlap, chunks = auto_chunk(documents, args.chunk_size, args.chunk_overlap)

    print("Generating English QA candidates...", flush=True)
    target_total = args.target_train + args.target_test
    qa_items = generate_qa(documents, chunks, target_total * 3)
    qa_counts = write_qa_splits(qa_items, args.target_train, args.target_test)

    stats = {
        "collected_at": collected_at,
        "document_count": len(documents),
        "failed_fetch_count": len(failures),
        "chunk_count": len(chunks),
        "chunk_size_words": chunk_size,
        "chunk_overlap_words": overlap,
        "qa_candidate_count": len(qa_items),
        "qa_counts": qa_counts,
        "line_validation": {
            "train_questions": count_lines(TRAIN_DIR / "questions.txt"),
            "train_answers": count_lines(TRAIN_DIR / "reference_answers.txt"),
            "test_questions": count_lines(TEST_DIR / "questions.txt"),
            "test_answers": count_lines(TEST_DIR / "reference_answers.txt"),
        },
        "failures_sample": failures[:50],
    }

    write_json(PROCESSED_DIR / "documents.json", documents)
    write_json(PROCESSED_DIR / "chunks.json", chunks)
    write_json(PROCESSED_DIR / "corpus_stats.json", stats)
    write_readmes(stats)

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Train QA: {qa_counts['train']}")
    print(f"Test QA: {qa_counts['test']}")
    print(f"Chunk size/overlap: {chunk_size}/{overlap}")

    if qa_counts["train"] < args.target_train or qa_counts["test"] < args.target_test:
        print("WARNING: target QA line counts not reached.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
