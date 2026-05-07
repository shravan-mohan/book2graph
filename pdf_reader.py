from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, NamedTuple
import re
import unicodedata

from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import LTChar, LTTextBox, LTTextLine
from pypdf import PdfReader, PdfWriter

COMMON_PROOF_COMPLETION_MARKERS: tuple[str, ...] = (
    "∎",
    "□",
    "◻",
    "◽",
    "▪",
    "■",
    "⬜",
    "⬛",
    "▢",
    "☐",
)


def iter_pdf_pages_text(pdf_path: str | Path) -> Iterator[tuple[int, str]]:
    """Yield (page_number, text) for each page in a readable PDF."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path}")

    reader = PdfReader(str(path))
    for page_index, page in enumerate(reader.pages, start=1):
        # extract_text() may return None on pages without extractable text.
        text = page.extract_text() or ""
        yield page_index, text


def detect_proof_completion_character(
    pdf_path: str | Path, markers: tuple[str, ...] = COMMON_PROOF_COMPLETION_MARKERS
) -> str | None:
    """
    Detect the most likely proof-completion symbol used in the PDF.

    Returns the most frequent matching marker, or None if no marker is found.
    """
    marker_counts: dict[str, int] = {marker: 0 for marker in markers}

    for _, page_text in iter_pdf_pages_text(pdf_path):
        for marker in markers:
            marker_counts[marker] += page_text.count(marker)

    # Fallback: some PDFs drop symbols in pypdf text extraction but keep them
    # when extracted with pdfminer.
    if not any(marker_counts.values()):
        raw_text = extract_text(str(Path(pdf_path)))
        for marker in markers:
            marker_counts[marker] += raw_text.count(marker)

    used_markers = {marker: count for marker, count in marker_counts.items() if count > 0}
    if not used_markers:
        return None

    return max(used_markers, key=used_markers.get)


class DrawnSquareMarker(NamedTuple):
    page_number: int
    x: float
    y: float
    width: float
    height: float


# Matches one stroked line segment inside a save/restore block:
#   q  1 0 0 1 <tx> <ty> cm  []0 d 0 J <lw> w  0 0 m  <dx> <dy> l S  Q
_LINE_SEG_RE = re.compile(
    r"q\s+1 0 0 1\s+([\d.+-]+)\s+([\d.+-]+)\s+cm\s+"
    r"\[\]0 d 0 J\s+[\d.]+\s+w\s+0 0 m\s+([\d.+-]+)\s+([\d.+-]+)\s+l S\s+Q"
)


def detect_drawn_square_markers(
    pdf_path: str | Path,
    min_side: float = 3.0,
    max_side: float = 20.0,
    squareness_tol: float = 3.0,
) -> list[DrawnSquareMarker]:
    """
    Detect end-of-proof squares drawn as four stroked line segments in the PDF.

    Some LaTeX-generated PDFs render the QED tombstone as four individual lines
    rather than a text character or a filled rectangle.  This function scans
    every page's content stream for groups of four consecutive line segments
    that form a small open square and returns their page numbers and positions.
    """
    reader = PdfReader(str(Path(pdf_path)))
    results: list[DrawnSquareMarker] = []

    for page_index, page in enumerate(reader.pages, start=1):
        contents = page.get_contents()
        if contents is None:
            continue
        stream = contents.get_data().decode("latin-1", errors="replace")
        segments = _LINE_SEG_RE.findall(stream)

        # Walk consecutive groups of 4 segments and check if they form a square.
        for i in range(len(segments) - 3):
            group = segments[i : i + 4]
            # Separate vertical (dx≈0) and horizontal (dy≈0) lines.
            verticals = [(tx, ty, dx, dy) for tx, ty, dx, dy in
                         ((float(tx), float(ty), float(dx), float(dy))
                          for tx, ty, dx, dy in group)
                         if abs(float(dx)) < 0.1 and abs(float(dy)) > min_side]
            horizontals = [(tx, ty, dx, dy) for tx, ty, dx, dy in
                           ((float(tx), float(ty), float(dx), float(dy))
                            for tx, ty, dx, dy in group)
                           if abs(float(dy)) < 0.1 and abs(float(dx)) > min_side]
            if len(verticals) != 2 or len(horizontals) != 2:
                continue

            height = abs(verticals[0][3])
            width = abs(horizontals[0][2])

            if not (min_side <= width <= max_side and min_side <= height <= max_side):
                continue
            if abs(width - height) > squareness_tol:
                continue

            # Position is the bottom-left translation of the first segment.
            tx0, ty0 = float(group[0][0]), float(group[0][1])
            results.append(DrawnSquareMarker(page_index, tx0, ty0, width, height))

    return results


def find_extracted_symbol_candidates(
    pdf_path: str | Path, min_count: int = 2
) -> list[tuple[str, int]]:
    """
    Return non-alphanumeric extracted symbols with their frequencies.

    This is useful when the proof marker is encoded as an unexpected glyph.
    """
    symbol_counts: dict[str, int] = {}

    def is_candidate_symbol(char: str) -> bool:
        if char.isspace() or char.isalnum():
            return False
        # Prioritize non-ASCII symbols and private-use glyphs.
        if ord(char) > 127:
            return True
        # Keep uncommon ASCII symbols that may appear as extracted markers.
        return char in {"*", "#"}

    for _, page_text in iter_pdf_pages_text(pdf_path):
        for char in page_text:
            if not is_candidate_symbol(char):
                continue
            symbol_counts[char] = symbol_counts.get(char, 0) + 1

    # pdfminer fallback to capture symbols that pypdf might miss.
    raw_text = extract_text(str(Path(pdf_path)))
    for char in raw_text:
        if not is_candidate_symbol(char):
            continue
        symbol_counts[char] = symbol_counts.get(char, 0) + 1

    filtered = [(char, count) for char, count in symbol_counts.items() if count >= min_count]
    # Symbols first, then private-use glyphs, then by frequency.
    return sorted(
        filtered,
        key=lambda item: (
            unicodedata.category(item[0]).startswith("P"),
            0xE000 <= ord(item[0]) <= 0xF8FF,
            -item[1],
        ),
    )


@dataclass
class MathBlock:
    """A numbered mathematical block (theorem, lemma, etc.) with its proof."""

    kind: str                         # "Theorem", "Lemma", …
    number: str                       # "1.4", "2.3"
    label: str                        # "Theorem 1.4"
    title: str | None                 # "(Characterization of Compact Sets)" or None
    statement_page_start: int
    statement_page_end: int           # last page before proof, or same as start
    proof_label: str | None           # "Proof.", "First proof of Theorem 1.4." …
    proof_page_start: int | None
    proof_page_end: int | None        # page carrying the QED drawn square
    statement_text: str
    proof_text: str | None


# ── font helpers ──────────────────────────────────────────────────────────────

def _font_is_bold(fontname: str) -> bool:
    """True for Computer-Modern bold-extended fonts (CMBX*) and generic Bold."""
    upper = fontname.upper()
    return "CMBX" in upper or "BOLD" in upper


def _font_is_italic(fontname: str) -> bool:
    """True for CMTI*, CMIT*, or generic Italic/Oblique."""
    upper = fontname.upper()
    return "CMTI" in upper or "CMIT" in upper or "ITALIC" in upper or "OBLIQUE" in upper


def _line_font_flags(line: LTTextLine) -> tuple[bool, bool]:
    """Return (has_bold, has_italic) based on character fonts in a line."""
    has_bold = has_italic = False
    for ch in line:
        if not isinstance(ch, LTChar):
            continue
        fn = ch.fontname
        if _font_is_bold(fn):
            has_bold = True
        if _font_is_italic(fn):
            has_italic = True
        if has_bold and has_italic:
            break
    return has_bold, has_italic


# ── patterns ─────────────────────────────────────────────────────────────────

_LIGATURE_TABLE = str.maketrans({
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st",
})


def _normalize_ligatures(text: str) -> str:
    """Replace common PDF ligature glyphs with their ASCII equivalents."""
    return text.translate(_LIGATURE_TABLE)


_BLOCK_KINDS = (
    "Theorem", "Lemma", "Corollary", "Definition", "Proposition",
    "Remark", "Exercise", "Example", "Notation",
)
_BLOCK_HEADER_RE = re.compile(
    r"^(?P<kind>" + "|".join(_BLOCK_KINDS) + r")"
    r"\s+(?P<number>\d+(?:\.\d+)*)"
    r"(?:\s*\((?P<title>[^)]*)\))?"
    r"\s*[.:]?\s*",
    re.IGNORECASE,
)
_PROOF_HEADER_RE = re.compile(
    r"^(?P<label>"
    r"(?:First|Second|Third|Alternative)\s+[Pp]roof(?:\s+of\s+\w+\s+[\d.]+)?"
    r"|[Pp]roof(?:\s+of\s+\w+\s+[\d.]+)?"
    r")\s*[.:]",
    re.IGNORECASE,
)


# ── main extraction function ──────────────────────────────────────────────────

def extract_math_blocks(pdf_path: str | Path) -> list[MathBlock]:
    """
    Extract all numbered math blocks (theorems, lemmas, definitions, …) and
    their proofs from a text-based PDF.

    Detection strategy
    ------------------
    * Block headers are lines that contain at least one bold (CMBX) character
      and match the pattern  ``Kind Number (optional title)``.
    * Proof headers are lines that start with italic (CMTI) characters and
      match  ``[First/Second/…] Proof [of Kind Number]``.
    * Proof endings are detected via the drawn QED squares found by
      :func:`detect_drawn_square_markers`.

    Returns a list of :class:`MathBlock` objects in document order.
    """
    path = Path(pdf_path)

    # Pre-build ordered list of (page_number) for each drawn QED square.
    qed_pages: list[int] = [m.page_number for m in detect_drawn_square_markers(path)]
    qed_index = 0  # pointer advancing through qed_pages as proofs are closed

    blocks: list[MathBlock] = []

    # ── state machine variables ──
    current_block: MathBlock | None = None
    stmt_lines: list[str] = []
    proof_lines: list[str] = []
    in_proof = False

    def _close_proof(end_page: int) -> None:
        nonlocal current_block, proof_lines, in_proof
        if current_block and in_proof:
            current_block.proof_page_end = end_page
            current_block.proof_text = " ".join(proof_lines).strip()
        proof_lines = []
        in_proof = False

    def _close_block(end_page: int) -> None:
        nonlocal current_block, stmt_lines
        if current_block:
            _close_proof(end_page)
            if not in_proof:
                current_block.statement_text = " ".join(stmt_lines).strip()
            blocks.append(current_block)
        current_block = None
        stmt_lines = []

    last_page = 1
    for page_num, layout in enumerate(extract_pages(str(path)), start=1):
        last_page = page_num

        for elem in layout:
            if not isinstance(elem, LTTextBox):
                continue
            for line in elem:
                if not isinstance(line, LTTextLine):
                    continue
                raw_text = line.get_text().strip()
                if not raw_text:
                    continue
                # Normalize ligature glyphs (e.g. ﬁ → fi) before regex matching.
                text = _normalize_ligatures(raw_text)

                has_bold, has_italic = _line_font_flags(line)

                # ── detect new block header ────────────────────────────────
                if has_bold:
                    m = _BLOCK_HEADER_RE.match(text)
                    if m:
                        _close_block(page_num)
                        current_block = MathBlock(
                            kind=m.group("kind").capitalize(),
                            number=m.group("number"),
                            label=f"{m.group('kind').capitalize()} {m.group('number')}",
                            title=m.group("title"),
                            statement_page_start=page_num,
                            statement_page_end=page_num,
                            proof_label=None,
                            proof_page_start=None,
                            proof_page_end=None,
                            statement_text="",
                            proof_text=None,
                        )
                        stmt_lines = [raw_text]
                        in_proof = False
                        proof_lines = []
                        continue

                # ── detect proof header ────────────────────────────────────
                if has_italic and current_block and not in_proof:
                    m = _PROOF_HEADER_RE.match(text)
                    if m:
                        current_block.statement_page_end = page_num
                        current_block.statement_text = " ".join(stmt_lines).strip()
                        current_block.proof_label = m.group("label").strip()
                        current_block.proof_page_start = page_num
                        in_proof = True
                        proof_lines = [raw_text]
                        continue

                # ── accumulate body text ───────────────────────────────────
                if current_block:
                    if in_proof:
                        proof_lines.append(raw_text)
                    else:
                        stmt_lines.append(raw_text)

        # ── after all lines on this page: consume QED squares ─────────────
        # Each QED square on this page closes one open proof.
        while in_proof and qed_index < len(qed_pages) and qed_pages[qed_index] == page_num:
            _close_proof(page_num)
            qed_index += 1
        # Advance past any squares on earlier pages that were never consumed.
        while qed_index < len(qed_pages) and qed_pages[qed_index] < page_num:
            qed_index += 1

    # Close any block still open at end of document.
    _close_block(last_page)

    return blocks


def extract_math_pages_pdf(
    source_pdf: str | Path,
    output_pdf: str | Path,
    blocks: list[MathBlock] | None = None,
) -> list[int]:
    """
    Create a new PDF containing only pages that belong to a math block.

    Each block's statement pages and proof pages are included.  Pages are
    written in their original document order and each appears at most once,
    regardless of how many blocks share that page.

    Parameters
    ----------
    source_pdf  : path to the original PDF.
    output_pdf  : destination path for the extracted PDF.
    blocks      : pre-computed list from :func:`extract_math_blocks`.
                  If *None*, the blocks are extracted automatically.

    Returns
    -------
    Sorted list of 1-based page numbers that were included.
    """
    source_path = Path(source_pdf)
    output_path = Path(output_pdf)

    if blocks is None:
        blocks = extract_math_blocks(source_path)

    # Collect every page that belongs to at least one block.
    included: set[int] = set()
    for block in blocks:
        for page in range(block.statement_page_start, block.statement_page_end + 1):
            included.add(page)
        if block.proof_page_start is not None:
            end = block.proof_page_end or block.proof_page_start
            for page in range(block.proof_page_start, end + 1):
                included.add(page)

    sorted_pages = sorted(included)

    reader = PdfReader(str(source_path))
    writer = PdfWriter()
    for page_num in sorted_pages:
        writer.add_page(reader.pages[page_num - 1])   # PdfReader is 0-indexed

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as fh:
        writer.write(fh)

    return sorted_pages


@dataclass
class SectionEntry:
    """One entry from the PDF's table of contents."""
    title: str
    depth: int
    page_start: int          # 1-based, inclusive
    page_end: int            # 1-based, inclusive
    is_leaf: bool            # True when this entry has no sub-sections
    output_path: Path | None = None


def _outline_to_entries(
    items: list,
    reader: PdfReader,
    depth: int,
    entries: list[SectionEntry],
) -> None:
    """Recursively walk pypdf's outline list and populate *entries*."""
    i = 0
    while i < len(items):
        item = items[i]
        if isinstance(item, list):
            i += 1
            continue
        try:
            page = reader.get_destination_page_number(item) + 1   # 1-based
        except Exception:
            i += 1
            continue
        has_children = i + 1 < len(items) and isinstance(items[i + 1], list)
        entries.append(SectionEntry(
            title=item.title,
            depth=depth,
            page_start=page,
            page_end=0,          # filled in after full traversal
            is_leaf=not has_children,
        ))
        if has_children:
            _outline_to_entries(items[i + 1], reader, depth + 1, entries)
        i += 1


def _slugify(title: str) -> str:
    """Turn an arbitrary section title into a safe filename stem."""
    nfkd = unicodedata.normalize("NFKD", title)
    ascii_title = nfkd.encode("ascii", errors="ignore").decode()
    slug = re.sub(r"[^\w\s-]", "", ascii_title).strip()
    slug = re.sub(r"[\s/\\]+", "_", slug)
    return slug[:60] or "section"


_DEFAULT_EXCLUDE_TITLES: set[str] = {"Problems"}   # problem sets excluded by default


def split_pdf_by_sections(
    source_pdf: str | Path,
    output_dir: str | Path,
    leaf_only: bool = True,
    start_from_title: str | None = None,
    exclude_titles: set[str] | None = None,
) -> list[SectionEntry]:
    """
    Split a PDF into one file per (leaf) section based on its built-in outline.

    Parameters
    ----------
    source_pdf       : path to the original PDF.
    output_dir       : directory where section PDFs will be written (created if needed).
    leaf_only        : when True (default) only the deepest sub-sections are written.
    start_from_title : if given, skip every section whose top-level (depth-0) ancestor
                       comes *before* the first depth-0 entry whose title matches this
                       string (case-insensitive prefix match).  Useful for skipping
                       front-matter such as "Introduction".
    exclude_titles   : set of exact section titles to drop entirely (title and all its
                       descendants).  Defaults to
                       ``{"References", "Notation", "Index",
                          "The Lemma of Zorn", "Tychonoff's Theorem"}``.

    Returns
    -------
    List of :class:`SectionEntry` objects in document order.
    """
    source_path = Path(source_pdf)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if exclude_titles is None:
        exclude_titles = _DEFAULT_EXCLUDE_TITLES

    reader = PdfReader(str(source_path))
    total_pages = len(reader.pages)

    if not reader.outline:
        raise ValueError("PDF has no built-in outline/table of contents.")

    # ── 1. flatten outline into ordered SectionEntry list ──────────────────
    entries: list[SectionEntry] = []
    _outline_to_entries(reader.outline, reader, depth=0, entries=entries)

    # ── 2. assign page_end for every entry ─────────────────────────────────
    by_page = sorted(range(len(entries)), key=lambda i: entries[i].page_start)
    for rank, idx in enumerate(by_page):
        if rank + 1 < len(by_page):
            next_start = entries[by_page[rank + 1]].page_start
            entries[idx].page_end = max(entries[idx].page_start, next_start - 1)
        else:
            entries[idx].page_end = total_pages

    # ── 3. propagate depth-0 ancestor titles so we can filter by chapter ───
    # Walk in document order; track the current depth-0 ancestor.
    current_chapter: str = ""
    chapter_for: list[str] = []
    for entry in entries:
        if entry.depth == 0:
            current_chapter = entry.title
        chapter_for.append(current_chapter)

    # Determine the first depth-0 title to include.
    start_chapter: str | None = None
    if start_from_title:
        needle = start_from_title.lower()
        for entry in entries:
            if entry.depth == 0 and entry.title.lower().startswith(needle):
                start_chapter = entry.title
                break

    # ── 4. apply filters ───────────────────────────────────────────────────
    filtered: list[SectionEntry] = []
    past_start = start_chapter is None    # True from the beginning if no filter set
    for entry, ch in zip(entries, chapter_for):
        # Activate once we hit the requested starting chapter.
        if start_chapter and ch == start_chapter:
            past_start = True
        if not past_start:
            continue
        # Drop excluded chapters/sections (match on the section itself or its chapter).
        if entry.title in exclude_titles or ch in exclude_titles:
            continue
        if leaf_only and not entry.is_leaf:
            continue
        filtered.append(entry)

    # ── 5. write one PDF per section ───────────────────────────────────────
    written: list[SectionEntry] = []
    for idx, entry in enumerate(filtered, start=1):
        writer = PdfWriter()
        for page_num in range(entry.page_start, entry.page_end + 1):
            writer.add_page(reader.pages[page_num - 1])

        filename = f"{idx:03d}_d{entry.depth}_{_slugify(entry.title)}.pdf"
        out_path = out_dir / filename
        with open(out_path, "wb") as fh:
            writer.write(fh)

        entry.output_path = out_path
        written.append(entry)

    return written


if __name__ == "__main__":
    target_pdf = Path.home() / "Downloads" / "funcana.pdf"
    print(f"Reading PDF: {target_pdf}")

    total_pages = 0
    non_empty_pages = 0
    for page_number, page_text in iter_pdf_pages_text(target_pdf):
        total_pages = page_number
        if page_text.strip():
            non_empty_pages += 1

    print(f"Total pages read: {total_pages}")
    print(f"Pages with extractable text: {non_empty_pages}")

    proof_marker = detect_proof_completion_character(target_pdf)
    if proof_marker is None:
        print("Detected proof completion marker (text): none found")
        print("Scanning for drawn (vector) square markers ...")
        drawn = detect_drawn_square_markers(target_pdf)
        if drawn:
            print(f"Found {len(drawn)} drawn square markers across the PDF")
            print("First 5 occurrences:")
            for m in drawn[:5]:
                print(f"  page {m.page_number}  pos=({m.x:.1f}, {m.y:.1f})  size={m.width:.1f}×{m.height:.1f}")
        else:
            print("No drawn square markers found either.")
    else:
        print(f"Detected proof completion marker: {proof_marker}")

    drawn = detect_drawn_square_markers(target_pdf)
    print(f"\nDrawn square markers summary: {len(drawn)} found across {len({m.page_number for m in drawn})} pages")

    print("\n--- Extracting math blocks ---")
    math_blocks = extract_math_blocks(target_pdf)
    kind_counts: dict[str, int] = {}
    for b in math_blocks:
        kind_counts[b.kind] = kind_counts.get(b.kind, 0) + 1
    print(f"Total blocks found: {len(math_blocks)}")
    for kind, count in sorted(kind_counts.items()):
        print(f"  {kind}: {count}")
    print("\nFirst 5 blocks:")
    for b in math_blocks[:5]:
        pspan = (
            f"p{b.proof_page_start}–{b.proof_page_end}"
            if b.proof_page_start else "no proof"
        )
        print(
            f"  [{b.label}] stmt p{b.statement_page_start}–{b.statement_page_end}"
            f"  proof {pspan}"
            f"  title={b.title}"
        )

    sections_dir = target_pdf.parent / (target_pdf.stem + "_sections")
    print(f"\n--- Splitting by leaf sections → {sections_dir}/ ---")
    sections = split_pdf_by_sections(
        target_pdf,
        sections_dir,
        start_from_title="Foundations",   # first real chapter; skips Introduction
    )
    print(f"Wrote {len(sections)} section PDFs:")
    for s in sections:
        print(f"  [{s.page_start:>3}–{s.page_end:<3}] d{s.depth}  {s.title}  →  {s.output_path.name}")

    output_pdf = target_pdf.parent / (target_pdf.stem + "_math_pages.pdf")
    print(f"\n--- Building math-pages PDF → {output_pdf} ---")
    included_pages = extract_math_pages_pdf(target_pdf, output_pdf, blocks=math_blocks)
    print(f"Included {len(included_pages)} of 452 pages")
    print(f"Saved: {output_pdf}")
