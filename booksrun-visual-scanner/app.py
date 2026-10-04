import base64
import io
import os
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageOps
import pytesseract


APP_TITLE = "BookScout"
GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"
BOOKSRUN_PRICE_URL = "https://booksrun.com/api/price/sell/{isbn}"
BOOKSRUN_CART_URL = "https://booksrun.com/api/cart/sell/add/{isbn}:{condition}?afk={afk}"
OPEN_LIBRARY_URL = "https://openlibrary.org/isbn/{isbn}.json"

MAX_IMAGE_DIM = 1600
JPEG_QUALITY = 82
BOOKSRUN_PROGRESS_TARGET = 15.00
DEFAULT_THRESHOLD = 3.00
HTTP_TIMEOUT = (3.5, 8.0)


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 760px;
        padding-top: 1rem;
        padding-bottom: 5rem;
    }
    .book-card {
        border: 1px solid rgba(128,128,128,.28);
        border-radius: 14px;
        padding: 14px;
        margin: 10px 0;
        background: rgba(128,128,128,.045);
    }
    .book-good {
        border: 2px solid #2e8b57;
        background: rgba(46,139,87,.08);
    }
    .book-muted {
        opacity: .72;
    }
    .book-title {
        font-size: 1.08rem;
        font-weight: 700;
        line-height: 1.25;
    }
    .book-author {
        color: rgba(128,128,128,.95);
        margin-bottom: 7px;
    }
    .price-row {
        display: flex;
        justify-content: space-between;
        gap: 8px;
        margin-top: 8px;
    }
    .price-pill {
        flex: 1;
        text-align: center;
        border-radius: 10px;
        padding: 7px 4px;
        background: rgba(128,128,128,.09);
    }
    .price-label {
        font-size: .72rem;
        opacity: .75;
    }
    .price-value {
        font-size: 1rem;
        font-weight: 700;
    }
    .small-note {
        font-size: .78rem;
        opacity: .72;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def secret_value(name: str, default: str = "") -> str:
    value = os.getenv(name, "")
    if value:
        return value.strip()

    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default

    return str(value).strip() if value is not None else default


BOOKSRUN_API_KEY = secret_value("BOOKSRUN_API_KEY")
BOOKSRUN_AFK = secret_value("BOOKSRUN_AFK")


def clean_text(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "").strip()
    return value


def normalize_text(value: str) -> str:
    value = value.lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def token_similarity(a: str, b: str) -> float:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)

    if not a_norm or not b_norm:
        return 0.0

    sequence_score = SequenceMatcher(None, a_norm, b_norm).ratio()

    a_tokens = set(a_norm.split())
    b_tokens = set(b_norm.split())
    overlap_score = len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))

    return 0.62 * sequence_score + 0.38 * overlap_score


def isbn10_to_13(isbn10: str) -> Optional[str]:
    value = re.sub(r"[^0-9Xx]", "", isbn10 or "").upper()

    if len(value) != 10 or not re.fullmatch(r"\d{9}[\dX]", value):
        return None

    core = "978" + value[:9]
    total = sum(
        (1 if index % 2 == 0 else 3) * int(character)
        for index, character in enumerate(core)
    )
    check = (10 - (total % 10)) % 10

    return core + str(check)


def valid_isbn13(isbn: str) -> bool:
    value = re.sub(r"\D", "", isbn or "")

    if len(value) != 13:
        return False

    total = sum(
        (1 if index % 2 == 0 else 3) * int(character)
        for index, character in enumerate(value)
    )

    return total % 10 == 0


def extract_isbns(text: str) -> List[str]:
    found: List[str] = []

    pattern = (
        r"(?<!\d)(?:97[89][\s-]?)?"
        r"\d(?:[\s-]?\d){9,12}(?:[\s-]?[0-9Xx])?(?!\d)"
    )

    for raw in re.findall(pattern, text or ""):
        digits = re.sub(r"[^0-9Xx]", "", raw).upper()

        if len(digits) == 13 and valid_isbn13(digits):
            found.append(digits)
        elif len(digits) == 10:
            converted = isbn10_to_13(digits)
            if converted:
                found.append(converted)

    return list(dict.fromkeys(found))


def downsample_image(raw: bytes, max_dim: int = MAX_IMAGE_DIM) -> bytes:
    with Image.open(io.BytesIO(raw)) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")

        width, height = image.size
        scale = min(1.0, max_dim / max(width, height))

        if scale < 1.0:
            image = image.resize(
                (
                    max(1, int(width * scale)),
                    max(1, int(height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )

        output = io.BytesIO()
        image.save(
            output,
            format="JPEG",
            quality=JPEG_QUALITY,
            optimize=True,
        )

        return output.getvalue()


def image_from_bytes(raw: bytes) -> Image.Image:
    return Image.open(io.BytesIO(raw)).convert("RGB")


def preprocess_for_ocr(image: Image.Image, angle: int) -> np.ndarray:
    rotated = image.rotate(
        angle,
        expand=True,
        fillcolor="white",
    )

    array = np.array(rotated)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    gray = clahe.apply(gray)

    return gray


def ocr_lines(image: Image.Image) -> Tuple[List[Dict[str, Any]], str]:
    results: List[Dict[str, Any]] = []
    all_text: List[str] = []

    # Running OCR at all three orientations improves shelf-spine recognition
    # without requiring a cloud vision model.
    for angle in (0, 90, 270):
        gray = preprocess_for_ocr(image, angle)

        if max(gray.shape) > 1800:
            scale = 1800 / max(gray.shape)
            gray = cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )

        data = pytesseract.image_to_data(
            gray,
            config="--oem 3 --psm 11",
            output_type=pytesseract.Output.DICT,
        )

        groups: Dict[
            Tuple[int, int, int],
            List[Tuple[str, float, int, int, int, int]],
        ] = {}

        for index, raw_text in enumerate(data["text"]):
            text = clean_text(raw_text)

            try:
                confidence = float(data["conf"][index])
            except (TypeError, ValueError):
                confidence = -1

            if not text or confidence < 22:
                continue

            key = (
                int(data["block_num"][index]),
                int(data["par_num"][index]),
                int(data["line_num"][index]),
            )

            groups.setdefault(key, []).append(
                (
                    text,
                    confidence,
                    int(data["left"][index]),
                    int(data["top"][index]),
                    int(data["width"][index]),
                    int(data["height"][index]),
                )
            )

        for words in groups.values():
            words.sort(key=lambda item: item[2])

            line_text = clean_text(
                " ".join(item[0] for item in words)
            )

            if len(re.sub(r"[^A-Za-z]", "", line_text)) < 3:
                continue

            average_confidence = (
                sum(item[1] for item in words) / len(words)
            )

            x1 = min(item[2] for item in words)
            y1 = min(item[3] for item in words)
            x2 = max(item[2] + item[4] for item in words)
            y2 = max(item[3] + item[5] for item in words)

            results.append(
                {
                    "text": line_text,
                    "confidence": average_confidence,
                    "angle": angle,
                    "x": x1,
                    "y": y1,
                    "w": x2 - x1,
                    "h": y2 - y1,
                }
            )
            all_text.append(line_text)

    deduped: List[Dict[str, Any]] = []
    seen = set()

    for item in sorted(
        results,
        key=lambda value: (
            -value["confidence"],
            -len(value["text"]),
        ),
    ):
        key = normalize_text(item["text"])

        if not key or key in seen:
            continue

        seen.add(key)
        deduped.append(item)

    return deduped, "\n".join(item["text"] for item in deduped)


def looks_like_noise(line: str) -> bool:
    normalized = normalize_text(line)

    if len(normalized) < 3:
        return True

    if len(normalized.split()) > 14:
        return True

    alpha_count = len(re.sub(r"[^A-Za-z]", "", line))
    digit_count = len(re.sub(r"[^0-9]", "", line))

    if alpha_count < 3 and digit_count < 3:
        return True

    return False


def candidate_from_lines(lines: List[str]) -> List[Dict[str, str]]:
    usable = [
        clean_text(line)
        for line in lines
        if not looks_like_noise(line)
    ]

    candidates: List[Dict[str, str]] = []

    # Strongest pattern: "Title by Author".
    for index, line in enumerate(usable):
        by_match = re.search(
            r"\bby\s+(.+)$",
            line,
            flags=re.IGNORECASE,
        )

        if not by_match:
            continue

        author = clean_text(by_match.group(1))
        prefix = clean_text(line[:by_match.start()])

        if prefix:
            candidates.append(
                {
                    "title": prefix,
                    "author": author,
                }
            )
        elif index > 0:
            candidates.append(
                {
                    "title": usable[index - 1],
                    "author": author,
                }
            )

    # General fallback for covers and spines where "by" is absent.
    for index in range(len(usable)):
        for span in (1, 2, 3):
            if index + span > len(usable):
                continue

            title = clean_text(
                " ".join(usable[index:index + span])
            )

            if len(title) < 4 or len(title) > 110:
                continue

            if re.search(
                r"\b(isbn|copyright|edition|published|www\.|com)\b",
                title,
                flags=re.IGNORECASE,
            ):
                continue

            author = ""

            if index + span < len(usable):
                next_line = usable[index + span]

                if (
                    len(next_line.split()) <= 7
                    and re.search(
                        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b",
                        next_line,
                    )
                ):
                    author = next_line

            candidates.append(
                {
                    "title": title,
                    "author": author,
                }
            )

    unique: List[Dict[str, str]] = []
    seen = set()

    for candidate in candidates:
        key = (
            normalize_text(candidate["title"]),
            normalize_text(candidate["author"]),
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(candidate)

    unique.sort(
        key=lambda candidate: (
            0 if candidate["author"] else 1,
            abs(len(candidate["title"].split()) - 3),
            -len(candidate["title"]),
        )
    )

    return unique[:12]


def google_books_search(
    title: str,
    author: str,
) -> List[Dict[str, Any]]:
    queries = []

    if title and author:
        queries.append(
            f'intitle:"{title}" inauthor:"{author}"'
        )

    if title:
        queries.append(f'intitle:"{title}"')

    if author:
        queries.append(f'inauthor:"{author}"')

    candidates: List[Dict[str, Any]] = []

    session = requests.Session()
    session.headers.update(
        {"User-Agent": "BookScout/1.0"}
    )

    for query in queries[:2]:
        try:
            response = session.get(
                GOOGLE_BOOKS_URL,
                params={
                    "q": query,
                    "maxResults": 10,
                    "printType": "books",
                },
                timeout=HTTP_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            continue

        for item in payload.get("items", []):
            info = item.get("volumeInfo", {})
            identifiers = info.get(
                "industryIdentifiers",
                [],
            )

            isbn13s = [
                identifier.get("identifier", "")
                for identifier in identifiers
                if (
                    identifier.get("type") == "ISBN_13"
                    and valid_isbn13(
                        identifier.get("identifier", "")
                    )
                )
            ]

            if not isbn13s:
                for identifier in identifiers:
                    converted = isbn10_to_13(
                        identifier.get("identifier", "")
                    )
                    if converted:
                        isbn13s.append(converted)

            if not isbn13s:
                continue

            result_title = clean_text(
                info.get("title", "")
            )
            result_authors = ", ".join(
                info.get("authors", [])
            )

            score = 0.0

            if title:
                score += 0.72 * token_similarity(
                    title,
                    result_title,
                )

            if author and result_authors:
                score += 0.28 * token_similarity(
                    author,
                    result_authors,
                )

            candidates.append(
                {
                    "isbn": isbn13s[0],
                    "title": result_title,
                    "author": result_authors,
                    "score": score,
                    "publisher": clean_text(
                        info.get("publisher", "")
                    ),
                    "published_date": clean_text(
                        info.get("publishedDate", "")
                    ),
                    "page_count": info.get("pageCount"),
                    "language": info.get("language", ""),
                    "thumbnail": info.get(
                        "imageLinks",
                        {},
                    ).get("thumbnail", ""),
                }
            )

    best: Dict[str, Dict[str, Any]] = {}

    for item in candidates:
        existing = best.get(item["isbn"])

        if (
            existing is None
            or item["score"] > existing["score"]
        ):
            best[item["isbn"]] = item

    return sorted(
        best.values(),
        key=lambda item: item["score"],
        reverse=True,
    )[:6]


@st.cache_data(
    ttl=120,
    max_entries=500,
    show_spinner=False,
)
def booksrun_price(
    isbn: str,
    api_key: str,
) -> Dict[str, Any]:
    if not api_key:
        return {
            "status": "not_configured",
            "text": "BooksRun API key is not configured.",
        }

    url = BOOKSRUN_PRICE_URL.format(
        isbn=isbn,
    )

    try:
        response = requests.get(
            url,
            params={"key": api_key},
            timeout=HTTP_TIMEOUT,
            headers={"User-Agent": "BookScout/1.0"},
        )
        response.raise_for_status()
        payload = response.json()
    except (
        requests.RequestException,
        ValueError,
    ) as exc:
        return {
            "status": "error",
            "text": f"BooksRun request failed: {exc}",
        }

    result = payload.get("result", {})

    if result.get("status") != "success":
        return {
            "status": "error",
            "text": (
                result.get("text")
                or result.get("message")
                or "No buyback quote."
            ),
        }

    prices = result.get("text", {})

    return {
        "status": "success",
        "average": float(
            prices.get("Average", 0) or 0
        ),
        "good": float(
            prices.get("Good", 0) or 0
        ),
        "new": float(
            prices.get("New", 0) or 0
        ),
    }


@st.cache_data(
    ttl=86400,
    max_entries=2000,
    show_spinner=False,
)
def open_library_format(isbn: str) -> str:
    try:
        response = requests.get(
            OPEN_LIBRARY_URL.format(isbn=isbn),
            timeout=HTTP_TIMEOUT,
            headers={"User-Agent": "BookScout/1.0"},
        )

        if response.status_code != 200:
            return "Unknown"

        payload = response.json()
    except (
        requests.RequestException,
        ValueError,
    ):
        return "Unknown"

    physical = clean_text(
        payload.get("physical_format", "")
    )
    lower = physical.lower()

    if "hardcover" in lower:
        return "HC"

    if (
        "paperback" in lower
        or "softcover" in lower
        or "mass market" in lower
    ):
        return "PB"

    return physical[:18] if physical else "Unknown"


def google_books_isbn_lookup(
    isbn: str,
) -> Dict[str, Any]:
    try:
        response = requests.get(
            f"{GOOGLE_BOOKS_URL}/{isbn}",
            timeout=HTTP_TIMEOUT,
            headers={"User-Agent": "BookScout/1.0"},
        )

        if response.status_code != 200:
            return {}

        return response.json().get(
            "volumeInfo",
            {},
        )
    except (
        requests.RequestException,
        ValueError,
    ):
        return {}


def resolve_book(
    candidate: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    google_candidates = google_books_search(
        candidate["title"],
        candidate["author"],
    )

    if not google_candidates:
        return {
            "title": candidate["title"],
            "author": (
                candidate["author"]
                or "Unknown author"
            ),
            "isbn": "",
            "format": "Unknown",
            "confidence": 0.0,
            "quote": {
                "status": "error",
                "text": (
                    "No Google Books ISBN-13 match."
                ),
            },
        }

    checked: List[Dict[str, Any]] = []

    for google_candidate in google_candidates[:5]:
        quote = booksrun_price(
            google_candidate["isbn"],
            BOOKSRUN_API_KEY,
        )

        merged = dict(google_candidate)
        merged["quote"] = quote
        checked.append(merged)

    best_metadata_score = max(
        item["score"] for item in checked
    )
    score_window = max(
        0.10,
        best_metadata_score * 0.12,
    )

    plausible = [
        item
        for item in checked
        if item["score"] >= best_metadata_score - score_window
    ]

    quoted = [
        item
        for item in plausible
        if item["quote"].get("status") == "success"
    ]

    if quoted:
        best = max(
            quoted,
            key=lambda item: (
                item["score"],
                float(
                    item["quote"].get(
                        "good",
                        0,
                    ) or 0
                ),
            ),
        )
    else:
        best = max(
            checked,
            key=lambda item: item["score"],
        )

    isbn = best["isbn"]

    return {
        "title": (
            best["title"]
            or candidate["title"]
        ),
        "author": (
            best["author"]
            or candidate["author"]
            or "Unknown author"
        ),
        "isbn": isbn,
        "format": open_library_format(isbn),
        "confidence": best["score"],
        "quote": best["quote"],
        "publisher": best.get(
            "publisher",
            "",
        ),
        "published_date": best.get(
            "published_date",
            "",
        ),
        "thumbnail": best.get(
            "thumbnail",
            "",
        ),
    }


def parse_manual_isbns(
    text: str,
) -> List[str]:
    values = []

    for token in re.split(
        r"[\s,;]+",
        text or "",
    ):
        digits = re.sub(
            r"[^0-9Xx]",
            "",
            token,
        ).upper()

        if (
            len(digits) == 13
            and valid_isbn13(digits)
        ):
            values.append(digits)

        elif len(digits) == 10:
            converted = isbn10_to_13(digits)

            if converted:
                values.append(converted)

    return list(dict.fromkeys(values))


def lookup_isbn(
    isbn: str,
) -> Dict[str, Any]:
    info = google_books_isbn_lookup(isbn)

    return {
        "title": (
            clean_text(
                info.get("title", "")
            )
            or "Unknown title"
        ),
        "author": (
            ", ".join(
                info.get("authors", [])
            )
            or "Unknown author"
        ),
        "isbn": isbn,
        "format": open_library_format(isbn),
        "confidence": 1.0,
        "quote": booksrun_price(
            isbn,
            BOOKSRUN_API_KEY,
        ),
    }


def cart_url(
    isbn: str,
    condition: str,
) -> str:
    if not BOOKSRUN_AFK:
        return ""

    return BOOKSRUN_CART_URL.format(
        isbn=isbn,
        condition=condition.lower(),
        afk=BOOKSRUN_AFK,
    )


def render_book_card(
    book: Dict[str, Any],
    threshold: float,
) -> None:
    quote = book.get("quote", {})

    if quote.get("status") == "success":
        average = float(
            quote.get("average", 0)
        )
        good = float(
            quote.get("good", 0)
        )
        new = float(
            quote.get("new", 0)
        )

        best_payout = max(
            average,
            good,
            new,
        )
        viable = best_payout >= threshold

    else:
        average = 0.0
        good = 0.0
        new = 0.0
        best_payout = 0.0
        viable = False

    card_class = (
        "book-card book-good"
        if viable
        else "book-card book-muted"
    )

    title = book.get(
        "title",
        "Unknown title",
    )
    author = book.get(
        "author",
        "Unknown author",
    )
    isbn = (
        book.get("isbn", "")
        or "Not resolved"
    )
    book_format = book.get(
        "format",
        "Unknown",
    )
    confidence = float(
        book.get("confidence", 0)
    )

    st.markdown(
        f"""
        <div class="{card_class}">
            <div class="book-title">{title}</div>
            <div class="book-author">{author}</div>
            <div class="small-note">
                Format: {book_format}
                &nbsp; | &nbsp;
                ISBN-13: {isbn}
            </div>
            <div class="small-note">
                Match confidence: {confidence:.0%}
            </div>
            <div class="price-row">
                <div class="price-pill">
                    <div class="price-label">Average</div>
                    <div class="price-value">
                        ${average:.2f}
                    </div>
                </div>
                <div class="price-pill">
                    <div class="price-label">Good</div>
                    <div class="price-value">
                        ${good:.2f}
                    </div>
                </div>
                <div class="price-pill">
                    <div class="price-label">New</div>
                    <div class="price-value">
                        ${new:.2f}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if (
        viable
        and isbn
        and BOOKSRUN_AFK
    ):
        if good >= threshold:
            condition = "good"
        elif average >= threshold:
            condition = "average"
        else:
            condition = "new"

        url = cart_url(
            isbn,
            condition,
        )

        st.link_button(
            (
                "🛒 Add to BooksRun Cart "
                f"(${best_payout:.2f})"
            ),
            url,
            use_container_width=True,
        )

    elif quote.get("status") != "success":
        st.caption(
            quote.get(
                "text",
                "No buyback quote available.",
            )
        )

    elif (
        isbn
        and not BOOKSRUN_AFK
    ):
        st.caption(
            "Set BOOKSRUN_AFK to enable "
            "the one-tap cart link."
        )


CAMERA_HTML = """
<div class="camera-shell">
  <video id="preview"
         autoplay
         playsinline
         muted></video>

  <canvas id="canvas" hidden></canvas>

  <div class="camera-status"
       id="status">
    Starting rear camera...
  </div>

  <div class="camera-controls">
    <button id="capture"
            type="button">
      📸 Capture
    </button>

    <label class="file-button">
      📁 Choose / Camera
      <input id="file"
             type="file"
             accept="image/*"
             capture="environment">
    </label>
  </div>

  <div class="zoom-row"
       id="zoom-row"
       hidden>
    <span>Zoom</span>
    <input id="zoom"
           type="range"
           min="1"
           max="1"
           step="0.1"
           value="1">
  </div>
</div>
"""

CAMERA_CSS = """
.camera-shell {
    width: 100%;
    font-family: var(--st-font);
    color: var(--st-text-color);
}

#preview {
    width: 100%;
    max-height: 55vh;
    object-fit: contain;
    border-radius: 14px;
    background: #111;
}

.camera-status {
    font-size: .8rem;
    opacity: .75;
    padding: 6px 0;
}

.camera-controls {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-top: 8px;
}

.camera-controls button,
.file-button {
    border: 1px solid rgba(128,128,128,.35);
    border-radius: 10px;
    padding: 11px 8px;
    text-align: center;
    cursor: pointer;
    font-weight: 600;
    background: rgba(128,128,128,.08);
}

.file-button input {
    display: none;
}

.zoom-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 8px;
}

.zoom-row input {
    flex: 1;
}
"""

CAMERA_JS = """
export default function(component) {
  const {
    parentElement,
    setTriggerValue
  } = component;

  const video =
    parentElement.querySelector("#preview");

  const canvas =
    parentElement.querySelector("#canvas");

  const captureButton =
    parentElement.querySelector("#capture");

  const fileInput =
    parentElement.querySelector("#file");

  const status =
    parentElement.querySelector("#status");

  const zoomRow =
    parentElement.querySelector("#zoom-row");

  const zoom =
    parentElement.querySelector("#zoom");

  let stream = null;
  let track = null;

  function stopStream() {
    if (stream) {
      stream.getTracks().forEach(
        (currentTrack) =>
          currentTrack.stop()
      );
    }

    stream = null;
    track = null;
  }

  function emitBlob(blob) {
    const reader = new FileReader();

    reader.onload = () => {
      setTriggerValue(
        "photo",
        reader.result
      );
    };

    reader.readAsDataURL(blob);
  }

  function resizeAndEmit(
    source,
    width,
    height
  ) {
    const maxDim = 1600;

    const scale = Math.min(
      1,
      maxDim / Math.max(
        width,
        height
      )
    );

    canvas.width =
      Math.max(
        1,
        Math.round(width * scale)
      );

    canvas.height =
      Math.max(
        1,
        Math.round(height * scale)
      );

    const context =
      canvas.getContext(
        "2d",
        { alpha: false }
      );

    context.drawImage(
      source,
      0,
      0,
      canvas.width,
      canvas.height
    );

    canvas.toBlob(
      (blob) => {
        if (blob) {
          emitBlob(blob);
        }
      },
      "image/jpeg",
      0.82
    );
  }

  async function startCamera() {
    if (
      !navigator.mediaDevices
      || !navigator.mediaDevices.getUserMedia
    ) {
      status.textContent =
        "Camera API unavailable. "
        + "Use Choose / Camera below.";
      return;
    }

    try {
      stream =
        await navigator.mediaDevices
          .getUserMedia({
            video: {
              facingMode: {
                exact: "environment"
              },
              width: {
                ideal: 1920
              },
              height: {
                ideal: 1080
              }
            },
            audio: false
          });

      video.srcObject = stream;

      track =
        stream.getVideoTracks()[0];

      const capabilities =
        track.getCapabilities
          ? track.getCapabilities()
          : {};

      if (
        capabilities.zoom
        && capabilities.zoom.max
          > capabilities.zoom.min
      ) {
        zoomRow.hidden = false;

        zoom.min =
          capabilities.zoom.min;

        zoom.max =
          capabilities.zoom.max;

        zoom.step =
          capabilities.zoom.step
          || 0.1;

        zoom.value =
          capabilities.zoom.min;
      }

      status.textContent =
        "Rear camera ready. "
        + "Fill the frame with the books.";

    } catch (error) {
      status.textContent =
        "Rear camera unavailable. "
        + "Use Choose / Camera below.";
    }
  }

  captureButton.onclick = () => {
    if (
      !video.videoWidth
      || !video.videoHeight
    ) {
      status.textContent =
        "Camera is not ready yet.";
      return;
    }

    resizeAndEmit(
      video,
      video.videoWidth,
      video.videoHeight
    );

    status.textContent =
      "Photo captured.";
  };

  fileInput.onchange = () => {
    const file =
      fileInput.files
      && fileInput.files[0];

    if (!file) {
      return;
    }

    const image = new Image();

    image.onload = () => {
      resizeAndEmit(
        image,
        image.naturalWidth,
        image.naturalHeight
      );

      URL.revokeObjectURL(
        image.src
      );

      status.textContent =
        "Photo selected.";
    };

    image.src =
      URL.createObjectURL(file);
  };

  zoom.oninput = async () => {
    if (
      !track
      || !track.applyConstraints
    ) {
      return;
    }

    try {
      await track.applyConstraints({
        advanced: [
          {
            zoom:
              Number(zoom.value)
          }
        ]
      });
    } catch (error) {
      // Some browsers expose zoom
      // capabilities but do not allow
      // changing them from JavaScript.
    }
  };

  startCamera();

  return () => {
    stopStream();
  };
}
"""

camera_component = st.components.v2.component(
    name="bookscout_rear_camera",
    html=CAMERA_HTML,
    css=CAMERA_CSS,
    js=CAMERA_JS,
)


def initialize_state() -> None:
    st.session_state.setdefault(
        "books",
        [],
    )
    st.session_state.setdefault(
        "manual_books",
        [],
    )
    st.session_state.setdefault(
        "threshold",
        DEFAULT_THRESHOLD,
    )
    st.session_state.setdefault(
        "last_image_hash",
        None,
    )


initialize_state()

st.title("📚 BookScout")
st.caption(
    "Scan books, check live BooksRun buyback value, "
    "and decide what is worth picking up."
)

threshold = st.slider(
    "Minimum Target Payout ($)",
    min_value=0.0,
    max_value=25.0,
    value=float(
        st.session_state.threshold
    ),
    step=0.50,
    key="threshold",
)

tab_scan, tab_manual = st.tabs(
    [
        "📷 Scan Books",
        "⌨️ ISBN Lookup",
    ]
)

with tab_scan:
    st.info(
        "For shelves, aim the camera straight at the spines. "
        "For table lots, spread books apart enough that titles "
        "and authors are readable."
    )

    camera_result = camera_component(
        key="bookscout_camera",
        on_photo_change=lambda: None,
        width="stretch",
        height=470,
    )

    uploaded = st.file_uploader(
        "Or upload a book photo",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
        accept_multiple_files=False,
        max_upload_size=25,
        help=(
            "Photos are resized to 1600 px "
            "before OCR."
        ),
    )

    photo_bytes = None

    if getattr(
        camera_result,
        "photo",
        None,
    ):
        data_url = camera_result.photo

        if (
            isinstance(data_url, str)
            and "," in data_url
        ):
            try:
                photo_bytes = base64.b64decode(
                    data_url.split(
                        ",",
                        1,
                    )[1]
                )
            except (
                ValueError,
                TypeError,
            ):
                photo_bytes = None

    elif uploaded is not None:
        photo_bytes = uploaded.getvalue()

    if photo_bytes:
        try:
            normalized_bytes = downsample_image(
                photo_bytes
            )
            image = image_from_bytes(
                normalized_bytes
            )

            st.image(
                image,
                caption="Image being analyzed",
                use_container_width=True,
            )

        except Exception as exc:
            st.error(
                f"Could not read the image: {exc}"
            )

        else:
            image_hash = hash(
                normalized_bytes
            )

            if (
                st.session_state.last_image_hash
                != image_hash
            ):
                st.session_state.last_image_hash = (
                    image_hash
                )
                st.session_state.books = []

                with st.status(
                    "Reading books...",
                    expanded=True,
                ) as status:
                    lines, raw_ocr = ocr_lines(
                        image
                    )

                    status.write(
                        "OCR found "
                        f"{len(lines)} text regions."
                    )

                    isbn_hits = extract_isbns(
                        raw_ocr
                    )

                    if isbn_hits:
                        status.write(
                            "Found "
                            f"{len(isbn_hits)} "
                            "ISBN-like numbers."
                        )

                        detected = [
                            lookup_isbn(isbn)
                            for isbn in isbn_hits
                        ]

                    else:
                        candidates = candidate_from_lines(
                            [
                                item["text"]
                                for item in lines
                            ]
                        )

                        status.write(
                            "Built "
                            f"{len(candidates)} "
                            "title candidates."
                        )

                        detected = []

                        progress = st.progress(
                            0.0
                        )

                        for index, candidate in enumerate(
                            candidates
                        ):
                            result = resolve_book(
                                candidate
                            )

                            if result:
                                detected.append(
                                    result
                                )

                            progress.progress(
                                (index + 1)
                                / max(
                                    1,
                                    len(candidates),
                                )
                            )

                    deduped = []
                    seen_titles = set()

                    for book in detected:
                        key = (
                            normalize_text(
                                book.get(
                                    "title",
                                    "",
                                )
                            ),
                            book.get(
                                "isbn",
                                "",
                            ),
                        )

                        if key in seen_titles:
                            continue

                        seen_titles.add(key)
                        deduped.append(book)

                    st.session_state.books = (
                        deduped
                    )

                    st.session_state.raw_ocr = (
                        raw_ocr
                    )

                    status.update(
                        label=(
                            "Finished. "
                            f"{len(deduped)} "
                            "book(s) resolved."
                        ),
                        state="complete",
                    )

            if st.session_state.books:
                st.subheader("Results")

                viable_total = 0.0

                for book in st.session_state.books:
                    quote = book.get(
                        "quote",
                        {},
                    )

                    if (
                        quote.get("status")
                        == "success"
                    ):
                        best = max(
                            float(
                                quote.get(
                                    "average",
                                    0,
                                )
                            ),
                            float(
                                quote.get(
                                    "good",
                                    0,
                                )
                            ),
                            float(
                                quote.get(
                                    "new",
                                    0,
                                )
                            ),
                        )

                        if best >= threshold:
                            viable_total += best

                    render_book_card(
                        book,
                        threshold,
                    )

                st.divider()
                st.subheader(
                    "Buyback goal"
                )

                st.metric(
                    "Viable buyback total",
                    f"${viable_total:.2f}",
                )

                progress_value = min(
                    1.0,
                    viable_total
                    / BOOKSRUN_PROGRESS_TARGET,
                )

                st.progress(
                    progress_value
                )

                st.caption(
                    f"${viable_total:.2f} of "
                    f"${BOOKSRUN_PROGRESS_TARGET:.2f} "
                    "toward the $15.00 target."
                )

                if (
                    viable_total
                    >= BOOKSRUN_PROGRESS_TARGET
                ):
                    st.success(
                        "You have reached the "
                        "$15.00 target."
                    )
                else:
                    st.info(
                        f"${BOOKSRUN_PROGRESS_TARGET - viable_total:.2f} "
                        "more to reach $15.00."
                    )

                with st.expander(
                    "OCR diagnostics"
                ):
                    st.text_area(
                        "Recognized text",
                        st.session_state.get(
                            "raw_ocr",
                            "",
                        ),
                        height=180,
                    )

    if not BOOKSRUN_API_KEY:
        st.warning(
            "BooksRun is not configured yet. "
            "Add BOOKSRUN_API_KEY in Streamlit Secrets "
            "to retrieve live payouts."
        )
    elif not BOOKSRUN_AFK:
        st.caption(
            "Live prices are enabled. "
            "Add BOOKSRUN_AFK to enable one-tap "
            "cart links."
        )

with tab_manual:
    st.subheader(
        "Manual ISBN lookup"
    )

    manual_text = st.text_area(
        "Paste one or more ISBN-10 or ISBN-13 values",
        height=140,
        placeholder=(
            "9780140449136\n"
            "9780061120084"
        ),
    )

    if st.button(
        "Look up ISBNs",
        type="primary",
        use_container_width=True,
    ):
        isbns = parse_manual_isbns(
            manual_text
        )

        if not isbns:
            st.error(
                "No valid ISBN-10 or ISBN-13 "
                "values were found."
            )
        else:
            results = []
            progress = st.progress(
                0.0
            )

            for index, isbn in enumerate(isbns):
                results.append(
                    lookup_isbn(isbn)
                )

                progress.progress(
                    (index + 1)
                    / len(isbns)
                )

            st.session_state.manual_books = (
                results
            )

    if st.session_state.manual_books:
        manual_total = 0.0

        for book in st.session_state.manual_books:
            quote = book.get(
                "quote",
                {},
            )

            if (
                quote.get("status")
                == "success"
            ):
                best = max(
                    float(
                        quote.get(
                            "average",
                            0,
                        )
                    ),
                    float(
                        quote.get(
                            "good",
                            0,
                        )
                    ),
                    float(
                        quote.get(
                            "new",
                            0,
                        )
                    ),
                )

                if best >= threshold:
                    manual_total += best

            render_book_card(
                book,
                threshold,
            )

        st.divider()

        st.metric(
            "Viable total",
            f"${manual_total:.2f}",
        )

        st.progress(
            min(
                1.0,
                manual_total
                / BOOKSRUN_PROGRESS_TARGET,
            )
        )

        st.caption(
            f"${manual_total:.2f} of "
            f"${BOOKSRUN_PROGRESS_TARGET:.2f}"
        )
