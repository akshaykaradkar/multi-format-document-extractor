"""
Hybrid Document Processing Pipeline

Supports 3 extraction modes:
1. rule_based - Fast, free, works for structured documents
2. ai_only    - Uses AI models for all documents (slower, costs)
3. hybrid     - Intelligent routing: rule-based first, AI fallback

This demonstrates Senior AI Engineer judgment:
- Knowing WHEN to use AI vs rule-based
- Cost-benefit analysis in architecture decisions
- Intelligent resource allocation

Author: Akshay Karadkar
"""

import json
import time
from pathlib import Path
from typing import Union, Optional, Literal
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from .config import get_file_type, OUTPUT_DIR, validate_config, OPENAI_API_KEY
from .schemas import RawExtraction, StandardizedOrder
from .parsers import PDFParser, ExcelParser, WordParser, CSVParser, OCRParser
from .parsers.base_parser import ParseError
from .processors import DataTransformer, ConfidenceScorer
from .processors.llm_extractor import LLMExtractor


class ExtractionMode(Enum):
    """Available extraction modes."""
    RULE_BASED = "rule_based"  # Fast, free, structured docs only
    AI_ONLY = "ai_only"        # AI for everything (Docling)
    HYBRID = "hybrid"          # Smart routing: rule-based + AI fallback


@dataclass
class ExtractionMetrics:
    """Metrics from extraction for comparison."""
    mode: str
    processing_time_ms: float
    confidence: float
    fields_extracted: int
    total_fields: int
    method_used: str
    estimated_cost: float  # USD
    success: bool
    error: Optional[str] = None


class HybridPipeline:
    """
    Intelligent document processing pipeline with 3 modes.

    This is what a Senior AI Engineer builds:
    - Not "AI everywhere" (wasteful)
    - Not "rule-based only" (brittle)
    - Intelligent routing based on document characteristics
    """

    # Cost estimates (USD per document)
    COSTS = {
        "rule_based": 0.0,
        "docling_ai": 0.005,  # Local AI model inference
        "gpt4_vision": 0.02,  # API call estimate
    }

    REQUIRED_FIELDS = [
        "order_id", "client_name", "order_date",
        "delivery_date", "items", "order_total"
    ]

    def __init__(
        self,
        mode: ExtractionMode = ExtractionMode.HYBRID,
        confidence_threshold: float = 0.7,
        verbose: bool = True,
    ):
        """
        Initialize pipeline.

        Args:
            mode: Extraction mode (rule_based, ai_only, hybrid)
            confidence_threshold: Below this, hybrid mode escalates to AI
            verbose: Print progress
        """
        self.mode = mode
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose

        # Initialize components
        self.transformer = DataTransformer()
        self.scorer = ConfidenceScorer()

        # Rule-based parsers
        self.parsers = {
            "pdf": PDFParser,
            "excel": ExcelParser,
            "word": WordParser,
            "csv": CSVParser,
            "ocr": OCRParser,
        }

        # AI extractor (lazy loaded)
        self._ai_extractor = None

        # LLM extractor for semantic parsing (when OpenAI key available)
        self._llm_extractor = None
        if OPENAI_API_KEY:
            try:
                self._llm_extractor = LLMExtractor()
            except Exception:
                pass

    @property
    def ai_extractor(self):
        """Lazy load AI extractor to avoid startup cost."""
        if self._ai_extractor is None:
            try:
                from .ai_models.docling_extractor import DoclingExtractor
                self._ai_extractor = DoclingExtractor()
            except ImportError as e:
                if self.verbose:
                    print(f"[WARN] AI extractor not available: {e}")
                self._ai_extractor = False  # Mark as unavailable
        return self._ai_extractor

    def process(
        self,
        file_path: Union[str, Path],
        save_output: bool = True,
    ) -> dict:
        """
        Process document with selected mode.

        Args:
            file_path: Path to document
            save_output: Save JSON output

        Returns:
            Dict with extraction results and metrics
        """
        file_path = Path(file_path)
        start_time = time.time()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Processing: {file_path.name}")
            print(f"Mode: {self.mode.value}")
            print(f"{'='*60}")

        try:
            if self.mode == ExtractionMode.RULE_BASED:
                result, metrics = self._extract_rule_based(file_path)
            elif self.mode == ExtractionMode.AI_ONLY:
                result, metrics = self._extract_ai_only(file_path)
            else:  # HYBRID
                result, metrics = self._extract_hybrid(file_path)

            # Calculate total time
            metrics.processing_time_ms = (time.time() - start_time) * 1000

            # Transform to standardized format
            if result:
                confidence = self.scorer.calculate_score(
                    result, validated=True, validation_errors=[]
                )
                standardized = self.transformer.transform(result, confidence)
                confidence_status = self.scorer.get_confidence_status(confidence)

                # Update metrics
                metrics.confidence = confidence
                metrics.fields_extracted = self._count_fields(result)
                metrics.success = True

                # Save output
                output_file = None
                if save_output:
                    output_file = self._save_output(standardized, file_path, metrics)

                if self.verbose:
                    self._print_metrics(metrics)

                return {
                    "success": True,
                    "order": standardized.model_dump(),
                    "metrics": metrics.__dict__,
                    "confidence_status": confidence_status,
                    "output_file": str(output_file) if output_file else None,
                }
            else:
                metrics.success = False
                metrics.error = "Extraction returned no results"
                return {
                    "success": False,
                    "error": metrics.error,
                    "metrics": metrics.__dict__,
                }

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return {
                "success": False,
                "error": str(e),
                "metrics": {
                    "mode": self.mode.value,
                    "processing_time_ms": processing_time,
                    "success": False,
                    "error": str(e),
                },
            }

    def _extract_rule_based(
        self,
        file_path: Path,
    ) -> tuple[Optional[RawExtraction], ExtractionMetrics]:
        """
        Rule-based extraction using pdfplumber, pandas, etc.

        Fast, free, works for structured documents.
        """
        metrics = ExtractionMetrics(
            mode="rule_based",
            processing_time_ms=0,
            confidence=0,
            fields_extracted=0,
            total_fields=len(self.REQUIRED_FIELDS),
            method_used="",
            estimated_cost=self.COSTS["rule_based"],
            success=False,
        )

        file_type = get_file_type(file_path)

        if self.verbose:
            print(f"[Rule-Based] Detected format: {file_type}")

        # Check if we have a parser
        if file_type == "ocr":
            # Rule-based can't handle scanned images
            if self.verbose:
                print("[Rule-Based] Cannot process scanned images")
            metrics.error = "Rule-based cannot process scanned/handwritten documents"
            return None, metrics

        parser_class = self.parsers.get(file_type)
        if not parser_class:
            metrics.error = f"No parser for format: {file_type}"
            return None, metrics

        # Parse document
        parser = parser_class(file_path)
        result = parser.parse()

        metrics.method_used = result.extraction_method
        metrics.success = True

        if self.verbose:
            print(f"[Rule-Based] Extracted with: {result.extraction_method}")

        return result, metrics

    def _extract_ai_only(
        self,
        file_path: Path,
    ) -> tuple[Optional[RawExtraction], ExtractionMetrics]:
        """
        AI-only extraction using Docling.

        Uses AI models for all documents regardless of complexity.
        """
        metrics = ExtractionMetrics(
            mode="ai_only",
            processing_time_ms=0,
            confidence=0,
            fields_extracted=0,
            total_fields=len(self.REQUIRED_FIELDS),
            method_used="docling_ai",
            estimated_cost=self.COSTS["docling_ai"],
            success=False,
        )

        if self.verbose:
            print("[AI-Only] Using IBM Docling for extraction...")

        if not self.ai_extractor:
            metrics.error = "AI extractor not available"
            return None, metrics

        # Use Docling
        ai_result = self.ai_extractor.extract(file_path)

        # Convert to RawExtraction format
        result = self._ai_result_to_raw(ai_result, file_path)

        metrics.method_used = ai_result.extraction_method
        metrics.confidence = ai_result.confidence
        metrics.success = True

        if self.verbose:
            print(f"[AI-Only] Docling confidence: {ai_result.confidence:.2f}")

        return result, metrics

    def _extract_hybrid(
        self,
        file_path: Path,
    ) -> tuple[Optional[RawExtraction], ExtractionMetrics]:
        """
        Hybrid extraction: rule-based first, AI fallback.

        This is the intelligent approach:
        1. Try rule-based (fast, free)
        2. Check confidence
        3. If low confidence OR scanned doc, use AI
        """
        file_type = get_file_type(file_path)

        if self.verbose:
            print(f"[Hybrid] Analyzing document: {file_type}")

        # Decision: Can rule-based handle this?
        needs_ai = self._needs_ai(file_path, file_type)

        if needs_ai:
            if self.verbose:
                print(f"[Hybrid] -> Routing to AI (reason: {needs_ai})")
            return self._extract_ai_only(file_path)

        # Try rule-based first
        if self.verbose:
            print("[Hybrid] -> Trying rule-based first...")

        result, metrics = self._extract_rule_based(file_path)

        if not result:
            # Rule-based failed, try AI
            if self.verbose:
                print("[Hybrid] -> Rule-based failed, escalating to AI")
            return self._extract_ai_only(file_path)

        # Check confidence
        temp_confidence = self.scorer.calculate_score(
            result, validated=True, validation_errors=[]
        )

        if temp_confidence < self.confidence_threshold:
            if self.verbose:
                print(f"[Hybrid] -> Low confidence ({temp_confidence:.2f}), escalating to AI")
            return self._extract_ai_only(file_path)

        # Rule-based succeeded with good confidence
        if self.verbose:
            print(f"[Hybrid] -> Rule-based succeeded (confidence: {temp_confidence:.2f})")

        metrics.mode = "hybrid_rule_based"
        return result, metrics

    def _needs_ai(self, file_path: Path, file_type: str) -> Optional[str]:
        """
        Determine if document needs AI processing.

        Returns reason string if AI needed, None otherwise.
        """
        # Scanned images always need AI
        if file_type == "ocr":
            return "scanned/handwritten document"

        # Check file characteristics (could be enhanced with image analysis)
        suffix = file_path.suffix.lower()
        if suffix in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            return "image file format"

        # Could add more heuristics:
        # - File size anomalies
        # - Previous failures from this client
        # - Document complexity analysis

        return None

    def _ai_result_to_raw(self, ai_result, file_path: Path) -> RawExtraction:
        """Convert Docling result to RawExtraction format."""
        fields = ai_result.fields
        markdown = ai_result.markdown

        # Try regex-based parsing first
        items = self._parse_items_from_markdown(markdown)
        if not items:
            items = self._parse_items_from_unstructured(markdown)

        order_total = fields.get("order_total")
        if not order_total:
            order_total = self._extract_total_from_markdown(markdown)

        order_date = fields.get("order_date")
        if not order_date:
            order_date = self._extract_date_from_markdown(markdown)

        delivery_date = fields.get("delivery_date")
        if not delivery_date:
            delivery_date = self._extract_delivery_date_from_markdown(markdown)

        client_name = fields.get("client_name")
        if not client_name:
            client_name = self._extract_client_from_markdown(markdown)

        order_id = fields.get("order_id")
        if order_id in ["FORM", "ORDER", None]:  # Common OCR misreads
            order_id = self._generate_order_id(file_path)

        # Build initial result
        result = RawExtraction(
            order_id=order_id,
            client_name=client_name,
            order_date=order_date,
            delivery_date=delivery_date,
            items=items,
            order_total=order_total,
            currency="USD",
            special_instructions=fields.get("special_instructions"),
            source_confidence=ai_result.confidence,
            extraction_method=ai_result.extraction_method,
        )

        # If regex parsing produced incomplete results and we have LLM, use it
        # This is the key for handling ANY scanned document format!
        if self._llm_extractor and self._is_incomplete(result):
            if self.verbose:
                print("[AI] Regex parsing incomplete, using LLM for semantic extraction...")
            try:
                result = self._llm_extractor.extract_from_text(markdown)
                result.extraction_method = "docling_ai+llm"
            except Exception as e:
                if self.verbose:
                    print(f"[AI] LLM extraction failed: {e}")

        return result

    def _is_incomplete(self, result: RawExtraction) -> bool:
        """Check if extraction result is incomplete."""
        # Consider incomplete if missing items or key fields
        if not result.items:
            return True
        if not result.order_id or result.order_id.startswith("OCR-"):
            return True
        if not result.client_name:
            return True
        return False

    def _parse_items_from_markdown(self, markdown: str) -> list[dict]:
        """Parse line items from markdown table format."""
        import re
        items = []

        # Find markdown table rows: | col1 | col2 | col3 | ...
        # Skip header separator rows (|---|---|...)
        table_row_pattern = r'\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|'

        for match in re.finditer(table_row_pattern, markdown):
            cells = [cell.strip() for cell in match.groups()]

            # Skip header row and separator rows
            if any(c.lower() in ['item code', 'description', 'qty', 'sku', 'product'] for c in cells):
                continue
            if any('---' in c for c in cells):
                continue
            if cells[0] == '' and 'TOTAL' in cells[3].upper():
                continue  # Skip total row

            # Parse row data
            try:
                # Typical format: code, description, qty, unit_price, total
                code = cells[0].strip()
                desc = cells[1].strip()
                qty_str = cells[2].strip()
                unit_price_str = cells[3].strip()
                total_str = cells[4].strip()

                if not desc or not qty_str:
                    continue

                # Parse numbers
                qty = int(re.sub(r'[^\d]', '', qty_str)) if qty_str else 0
                unit_price = float(re.sub(r'[^\d.]', '', unit_price_str)) if unit_price_str else 0.0
                total_price = float(re.sub(r'[^\d.]', '', total_str)) if total_str else qty * unit_price

                if desc and qty > 0:
                    items.append({
                        "product_code": code if code else self._generate_code(desc),
                        "description": desc,
                        "quantity": qty,
                        "unit_price": unit_price,
                        "total_price": total_price,
                    })
            except (ValueError, IndexError):
                continue

        return items

    def _extract_total_from_markdown(self, markdown: str) -> float | None:
        """Extract order total from markdown content."""
        import re
        patterns = [
            r'TOTAL:?\s*\$?([\d,]+\.?\d*)',
            r'\|\s*TOTAL:?\s*\|\s*\$?([\d,]+\.?\d*)\s*\|',
        ]
        for pattern in patterns:
            match = re.search(pattern, markdown, re.IGNORECASE)
            if match:
                return float(match.group(1).replace(',', ''))
        return None

    def _generate_code(self, description: str) -> str:
        """Generate product code from description."""
        words = description.split()[:2]
        return "-".join(w[:3].upper() for w in words) if words else "ITEM"

    def _parse_items_from_unstructured(self, markdown: str) -> list[dict]:
        """Parse items from unstructured/handwritten format."""
        import re
        items = []

        # Pattern for "Item N: description" followed by "Qty: N" and optionally "Price: $X"
        # Example: "Item 1:\nScrews\nQty:\n6\nPrice: $28.00"
        item_pattern = r'Item\s*\d+:?\s*\n?([^\n]+)\s*\n?Qty:?\s*\n?(\d+)'
        price_pattern = r'Price:?\s*\$?([\d.,]+)'

        lines = markdown.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Look for "Item N:" pattern
            if re.match(r'Item\s*\d+:?', line, re.IGNORECASE):
                desc = ""
                qty = 0
                price = 0.0

                # Next non-empty line is description
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines):
                    desc = lines[j].strip().replace('_', ' ')

                # Look for Qty in next few lines
                for k in range(j+1, min(j+5, len(lines))):
                    if 'qty' in lines[k].lower():
                        # Next line or same line has the number
                        qty_match = re.search(r'(\d+)', lines[k])
                        if qty_match:
                            qty = int(qty_match.group(1))
                        elif k+1 < len(lines):
                            qty_match = re.search(r'(\d+)', lines[k+1])
                            if qty_match:
                                qty = int(qty_match.group(1))
                        break

                # Look for Price
                for k in range(j+1, min(j+8, len(lines))):
                    if 'price' in lines[k].lower():
                        price_match = re.search(r'\$?([\d.,]+)', lines[k])
                        if price_match:
                            price = float(price_match.group(1).replace(',', ''))
                        break

                if desc and qty > 0:
                    items.append({
                        "product_code": self._generate_code(desc),
                        "description": desc,
                        "quantity": qty,
                        "unit_price": price,
                        "total_price": price * qty if price else 0.0,
                    })

            i += 1

        return items

    def _extract_date_from_markdown(self, markdown: str) -> str | None:
        """Extract order date from markdown."""
        import re
        from dateutil import parser as date_parser

        # Look for "Date:" followed by date
        patterns = [
            r'Date:?\s*\n?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',
            r'Date:?\s*\n?\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
        ]
        for pattern in patterns:
            match = re.search(pattern, markdown, re.IGNORECASE)
            if match:
                try:
                    parsed = date_parser.parse(match.group(1))
                    return parsed.strftime("%Y-%m-%d")
                except:
                    pass
        return None

    def _extract_delivery_date_from_markdown(self, markdown: str) -> str | None:
        """Extract delivery/need-by date from markdown."""
        import re
        from dateutil import parser as date_parser

        patterns = [
            r'Need\s*by:?\s*\n?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',
            r'Delivery:?\s*\n?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',
        ]
        for pattern in patterns:
            match = re.search(pattern, markdown, re.IGNORECASE)
            if match:
                try:
                    date_str = match.group(1)
                    # Fix common OCR errors like "3/2512024" -> "3/25/2024"
                    date_str = re.sub(r'(\d{1,2})/(\d{2})(\d{4})', r'\1/\2/\3', date_str)
                    parsed = date_parser.parse(date_str)
                    return parsed.strftime("%Y-%m-%d")
                except:
                    pass
        return None

    def _extract_client_from_markdown(self, markdown: str) -> str | None:
        """Extract client name from markdown header."""
        import re

        # Look for header lines that might be company names
        lines = markdown.split('\n')
        for line in lines[:10]:
            line = line.strip()
            # Match "## COMPANY NAME" or "LOCAL HARDWARE" etc
            if line.startswith('##'):
                name = line.replace('#', '').strip()
                if name and 'ORDER' not in name.upper() and 'FORM' not in name.upper():
                    return name

            # Common business suffixes
            if any(s in line.upper() for s in ['HARDWARE', 'SUPPLY', 'INC', 'LLC', 'CORP']):
                return line.replace('#', '').strip()

        return None

    def _generate_order_id(self, file_path: Path) -> str:
        """Generate order ID from filename when OCR fails."""
        from datetime import datetime
        stem = file_path.stem.replace('client_', '').replace('_', '-')
        return f"OCR-{stem[:10].upper()}-{datetime.now().strftime('%H%M')}"

    def _count_fields(self, result: RawExtraction) -> int:
        """Count number of populated fields."""
        count = 0
        if result.order_id:
            count += 1
        if result.client_name:
            count += 1
        if result.order_date:
            count += 1
        if result.delivery_date:
            count += 1
        if result.items:
            count += 1
        if result.order_total:
            count += 1
        return count

    def _save_output(
        self,
        order: StandardizedOrder,
        source_file: Path,
        metrics: ExtractionMetrics,
    ) -> Path:
        """Save output with metrics."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{source_file.stem}_{self.mode.value}_{timestamp}.json"
        output_path = OUTPUT_DIR / output_name

        output_data = {
            "order": order.model_dump(),
            "extraction_metrics": {
                "mode": metrics.mode,
                "processing_time_ms": metrics.processing_time_ms,
                "confidence": metrics.confidence,
                "method_used": metrics.method_used,
                "estimated_cost_usd": metrics.estimated_cost,
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        return output_path

    def _print_metrics(self, metrics: ExtractionMetrics):
        """Print extraction metrics."""
        print(f"\n{'-'*60}")
        print(f"Extraction Metrics:")
        print(f"  Mode:        {metrics.mode}")
        print(f"  Method:      {metrics.method_used}")
        print(f"  Time:        {metrics.processing_time_ms:.0f}ms")
        print(f"  Confidence:  {metrics.confidence:.2f}")
        print(f"  Fields:      {metrics.fields_extracted}/{metrics.total_fields}")
        print(f"  Est. Cost:   ${metrics.estimated_cost:.4f}")
        print(f"{'-'*60}")


def process_with_mode(
    file_path: Union[str, Path],
    mode: str = "hybrid",
    save_output: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Convenience function to process document with specified mode.

    Args:
        file_path: Path to document
        mode: "rule_based", "ai_only", or "hybrid"
        save_output: Save JSON output
        verbose: Print progress

    Returns:
        Processing result dict
    """
    mode_enum = ExtractionMode(mode)
    pipeline = HybridPipeline(mode=mode_enum, verbose=verbose)
    return pipeline.process(file_path, save_output=save_output)
