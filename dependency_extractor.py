"""
dependency_extractor.py
-----------------------
Upload each section PDF to Gemini and ask it to identify which theorems,
lemmas, corollaries, and definitions each proof depends on.

Usage
-----
Fill in GEMINI_API_KEY (and optionally GEMINI_MODEL) in the .env file, then:

    from dependency_extractor import extract_dependencies_from_sections
    results = extract_dependencies_from_sections(section_pdfs)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# ── configuration ─────────────────────────────────────────────────────────────

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-04-17")

_SYSTEM_PROMPT = """\
You are a precise mathematical text analyser.
You will be given a section of a mathematics textbook as a PDF.
Your task is to identify every numbered mathematical block in it
(Theorem, Lemma, Corollary, Proposition, Definition, Example, Remark)
and, for each block that has a proof, list the other numbered blocks
whose results are explicitly used or cited in that proof.

Rules
-----
- Only include blocks that are directly cited by label in the proof text
  (e.g. "by Lemma 1.3", "from Theorem 2.4", "Corollary 1.8 gives …").
- Do NOT include blocks that are merely mentioned in the statement.
- If a proof cites no other numbered block, return an empty depends_on list.
- Use the exact label as it appears in the text (e.g. "Theorem 1.4",
  "Lemma 2.7", "Definition 3.1").
- For blocks without a proof (e.g. definitions, unreproved assertions),
  still include them but set has_proof to false and depends_on to [].

Return ONLY a JSON object with this exact schema — no markdown, no commentary:

{
  "section_title": "<title of this section if discernible, else null>",
  "blocks": [
    {
      "label":      "<Kind Number>",
      "kind":       "<Theorem|Lemma|Corollary|Proposition|Definition|Example|Remark>",
      "number":     "<e.g. 1.4>",
      "title":      "<parenthetical title if present, else null>",
      "has_proof":  true,
      "depends_on": ["<label>", ...]
    }
  ]
}
"""


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class BlockDependency:
    label: str
    kind: str
    number: str
    title: str | None
    has_proof: bool
    depends_on: list[str]


@dataclass
class SectionDependencies:
    section_title: str | None
    source_pdf: Path
    blocks: list[BlockDependency] = field(default_factory=list)
    error: str | None = None          # set if the API call failed


# ── core extraction ───────────────────────────────────────────────────────────

def _build_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Fill it in the .env file."
        )
    return genai.Client(api_key=api_key)


def extract_dependencies_from_pdf(
    pdf_path: str | Path,
    client: genai.Client | None = None,
    model: str = DEFAULT_MODEL,
    retry_delay: float = 5.0,
    max_retries: int = 3,
) -> SectionDependencies:
    """
    Upload *pdf_path* to Gemini, ask for theorem dependency data, and return
    a :class:`SectionDependencies` object.

    Parameters
    ----------
    pdf_path     : path to the section PDF.
    client       : a ``genai.Client`` instance; created from .env if omitted.
    model        : Gemini model name (override with GEMINI_MODEL in .env).
    retry_delay  : seconds to wait between retries on transient errors.
    max_retries  : number of retry attempts on rate-limit / server errors.
    """
    path = Path(pdf_path)
    if client is None:
        client = _build_client()

    result = SectionDependencies(section_title=None, source_pdf=path)

    # ── upload the PDF ────────────────────────────────────────────────────
    uploaded_file = client.files.upload(
        file=path,
        config=types.UploadFileConfig(mime_type="application/pdf"),
    )

    # ── prompt Gemini ─────────────────────────────────────────────────────
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_uri(
                        file_uri=uploaded_file.uri,
                        mime_type="application/pdf",
                    ),
                    _SYSTEM_PROMPT,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )
            break
        except Exception as exc:
            if attempt == max_retries:
                result.error = str(exc)
                return result
            time.sleep(retry_delay * attempt)

    # ── parse JSON response ───────────────────────────────────────────────
    try:
        raw = json.loads(response.text)
    except json.JSONDecodeError as exc:
        result.error = f"JSON parse error: {exc}\nRaw response: {response.text[:500]}"
        return result

    result.section_title = raw.get("section_title")
    for item in raw.get("blocks", []):
        result.blocks.append(BlockDependency(
            label=item.get("label", ""),
            kind=item.get("kind", ""),
            number=item.get("number", ""),
            title=item.get("title"),
            has_proof=bool(item.get("has_proof", False)),
            depends_on=item.get("depends_on", []),
        ))

    # ── clean up uploaded file to avoid storage accumulation ─────────────
    try:
        client.files.delete(name=uploaded_file.name)
    except Exception:
        pass

    return result


def extract_dependencies_from_sections(
    section_pdfs: list[Path],
    model: str = DEFAULT_MODEL,
    request_delay: float = 2.0,
    save_to: Path | None = None,
) -> list[SectionDependencies]:
    """
    Run :func:`extract_dependencies_from_pdf` on every section PDF in the list.

    Parameters
    ----------
    section_pdfs   : ordered list of PDF paths (e.g. from split_pdf_by_sections).
    model          : Gemini model to use.
    request_delay  : seconds to sleep between requests (avoids rate limits).
    save_to        : if given, write the combined results as JSON to this path
                     after every successful section (incremental save).

    Returns
    -------
    List of :class:`SectionDependencies` objects, one per input PDF.
    """
    client = _build_client()
    all_results: list[SectionDependencies] = []

    for i, pdf_path in enumerate(section_pdfs, start=1):
        print(f"[{i}/{len(section_pdfs)}] Processing: {pdf_path.name} …", flush=True)
        result = extract_dependencies_from_pdf(pdf_path, client=client, model=model)

        if result.error:
            print(f"  ✗ Error: {result.error}")
        else:
            dep_count = sum(len(b.depends_on) for b in result.blocks)
            print(
                f"  ✓ {len(result.blocks)} blocks, {dep_count} dependencies"
                + (f' \u2014 \u201c{result.section_title}\u201d' if result.section_title else "")
            )

        all_results.append(result)

        if save_to:
            _save_results(all_results, save_to)

        if i < len(section_pdfs):
            time.sleep(request_delay)

    return all_results


# ── serialisation helpers ─────────────────────────────────────────────────────

def _result_to_dict(result: SectionDependencies) -> dict:
    return {
        "source_pdf": str(result.source_pdf),
        "section_title": result.section_title,
        "error": result.error,
        "blocks": [
            {
                "label": b.label,
                "kind": b.kind,
                "number": b.number,
                "title": b.title,
                "has_proof": b.has_proof,
                "depends_on": b.depends_on,
            }
            for b in result.blocks
        ],
    }


def _save_results(results: list[SectionDependencies], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([_result_to_dict(r) for r in results], fh, indent=2, ensure_ascii=False)


def load_results(path: str | Path) -> list[dict]:
    """Load previously saved results JSON from *path*."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ── __main__ ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    sections_dir = Path.home() / "Downloads" / "funcana_sections"
    output_json  = Path.home() / "Downloads" / "funcana_dependencies.json"

    pdfs = sorted(sections_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {sections_dir}. Run pdf_reader.py first.")
        sys.exit(1)

    print(f"Found {len(pdfs)} section PDFs in {sections_dir}")
    print(f"Results will be saved incrementally to {output_json}\n")

    results = extract_dependencies_from_sections(
        pdfs,
        save_to=output_json,
    )

    total_blocks = sum(len(r.blocks) for r in results)
    total_deps   = sum(len(b.depends_on) for r in results for b in r.blocks)
    errors       = [r for r in results if r.error]

    print(f"\nDone.")
    print(f"  Sections processed : {len(results)}")
    print(f"  Blocks found       : {total_blocks}")
    print(f"  Dependencies found : {total_deps}")
    print(f"  Errors             : {len(errors)}")
    print(f"  Output JSON        : {output_json}")
