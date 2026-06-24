"""
PDF extraction pipeline using Docling.
Output: Markdown per PDF, max accuracy (OCR + table/layout preservation).
"""

from pathlib import Path
import logging
import sys
import time

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
    TesseractCliOcrOptions,
)
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

INPUT_DIR = Path(__file__).parent
OUTPUT_DIR = INPUT_DIR / "output_md"


def build_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        table_structure_options=TableStructureOptions(
            mode=TableFormerMode.ACCURATE,
            do_cell_matching=True,
        ),
        ocr_options=TesseractCliOcrOptions(
            lang=["deu", "eng"],        # dokumenty DE + EN (kody tesseract)
            force_full_page_ocr=False,  # OCR tylko tam gdzie brak tekstu wektorowego
        ),
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend,
            )
        }
    )


def extract_all(input_dir: Path, output_dir: Path) -> None:
    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        log.error("Brak plikow PDF w: %s", input_dir)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Znaleziono %d PDF(ow) -> wynik w: %s", len(pdfs), output_dir)

    converter = build_converter()
    errors: list[tuple[Path, Exception]] = []

    for i, pdf_path in enumerate(pdfs, 1):
        out_path = output_dir / (pdf_path.stem + ".md")
        log.info("[%d/%d] Przetwarzam: %s", i, len(pdfs), pdf_path.name)
        t0 = time.perf_counter()
        try:
            result = converter.convert(pdf_path)
            md_text = result.document.export_to_markdown(
                escape_html=False,
                escape_underscores=False,
                image_placeholder="",
            )
            out_path.write_text(md_text, encoding="utf-8")
            elapsed = time.perf_counter() - t0
            log.info("  OK -> %s (%.1fs, %d znakow)", out_path.name, elapsed, len(md_text))
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            log.error("  BLAD po %.1fs: %s -- %s", elapsed, pdf_path.name, exc)
            errors.append((pdf_path, exc))

    log.info("=" * 60)
    log.info("Gotowe: %d/%d sukces", len(pdfs) - len(errors), len(pdfs))
    if errors:
        log.error("Nieudane pliki:")
        for p, e in errors:
            log.error("  %s: %s", p.name, e)
        sys.exit(2)


if __name__ == "__main__":
    extract_all(INPUT_DIR, OUTPUT_DIR)
