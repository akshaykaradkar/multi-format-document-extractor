"""
IBM Docling-based Document Extractor

Production-ready AI document understanding using IBM's Docling library.
This is the 2024/2025 state-of-the-art approach for document processing.

Key AI Models Used:
- DocLayNet: Layout analysis (header, table, text regions)
- TableFormer: Table structure recognition
- EasyOCR: Text recognition for scanned documents
- Granite-Docling: Optional VLM for complex documents

Reference:
- GitHub: https://github.com/docling-project/docling
- Paper: "Docling: An Efficient Open-Source Toolkit for AI-driven Document Conversion" (AAAI 2025)
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field

# Suppress symlink warnings on Windows
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    HAS_DOCLING = True
except ImportError:
    HAS_DOCLING = False
    print("Warning: docling not installed. Install with: pip install docling")


@dataclass
class DoclingExtractionResult:
    """Result from Docling extraction."""
    fields: Dict[str, str] = field(default_factory=dict)
    tables: List[Dict] = field(default_factory=list)
    markdown: str = ""
    confidence: float = 0.0
    processing_time: float = 0.0
    extraction_method: str = "docling"


class DoclingExtractor:
    """
    AI-powered document extractor using IBM Docling.

    This is the recommended approach for production document processing.
    Uses state-of-the-art AI models for layout understanding and text extraction.
    """

    def __init__(
        self,
        enable_table_structure: bool = True,
        enable_ocr: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize Docling extractor.

        Args:
            enable_table_structure: Use TableFormer for table extraction
            enable_ocr: Use EasyOCR for scanned documents
            device: Device to run on ('cpu' or 'cuda')
        """
        if not HAS_DOCLING:
            raise ImportError("docling not installed. Run: pip install docling")

        self.enable_table_structure = enable_table_structure
        self.enable_ocr = enable_ocr
        self.device = device

        # Initialize converter
        self._init_converter()

    def _init_converter(self):
        """Initialize the Docling document converter."""
        # Configure pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = self.enable_table_structure
        pipeline_options.do_ocr = self.enable_ocr

        # Create converter with options
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def extract(
        self,
        file_path: Union[str, Path],
    ) -> DoclingExtractionResult:
        """
        Extract content from document using AI.

        Args:
            file_path: Path to document (PDF, DOCX, XLSX, etc.)

        Returns:
            DoclingExtractionResult with extracted fields and tables
        """
        import time
        start_time = time.time()

        file_path = Path(file_path)

        # Convert document
        result = self.converter.convert(str(file_path))

        # Export to markdown
        markdown = result.document.export_to_markdown()

        # Extract structured fields
        fields = self._extract_fields_from_markdown(markdown)

        # Extract tables
        tables = self._extract_tables(result.document)

        processing_time = time.time() - start_time

        # Calculate confidence based on extraction quality
        confidence = self._calculate_confidence(fields, tables)

        return DoclingExtractionResult(
            fields=fields,
            tables=tables,
            markdown=markdown,
            confidence=confidence,
            processing_time=processing_time,
            extraction_method="docling_ai",
        )

    def _extract_fields_from_markdown(self, markdown: str) -> Dict[str, str]:
        """
        Extract structured fields from markdown output.

        Uses regex patterns to find common document fields.
        This can be enhanced with LLM for better semantic understanding.
        """
        fields = {}

        # Order ID patterns
        order_patterns = [
            r'(?:PURCHASE ORDER|PO|Order)\s*#?\s*([A-Z0-9\-]+)',
            r'Order\s*(?:Number|No\.?|#|ID)\s*:?\s*([A-Z0-9\-]+)',
        ]
        for pattern in order_patterns:
            match = re.search(pattern, markdown, re.IGNORECASE)
            if match:
                fields['order_id'] = match.group(1).strip()
                break

        # Client/Company name (usually in header)
        lines = markdown.split('\n')
        for line in lines[:5]:
            line = line.strip()
            if line.startswith('## ') and any(
                suffix in line for suffix in ['Solutions', 'Inc', 'Corp', 'LLC', 'Ltd', 'Co']
            ):
                fields['client_name'] = line.replace('## ', '').strip()
                break

        # Dates
        date_patterns = [
            (r'Date:\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})', 'order_date'),
            (r'Order Date:\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})', 'order_date'),
            (r'Delivery\s*(?:Required|Date):\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})', 'delivery_date'),
            (r'Ship\s*(?:By|Date):\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})', 'delivery_date'),
        ]
        for pattern, field_name in date_patterns:
            match = re.search(pattern, markdown, re.IGNORECASE)
            if match and field_name not in fields:
                fields[field_name] = self._normalize_date(match.group(1))

        # Total amount
        total_patterns = [
            r'TOTAL:?\s*\$?([\d,]+\.?\d*)',
            r'Grand Total:?\s*\$?([\d,]+\.?\d*)',
            r'Order Total:?\s*\$?([\d,]+\.?\d*)',
        ]
        for pattern in total_patterns:
            match = re.search(pattern, markdown, re.IGNORECASE)
            if match:
                fields['order_total'] = float(match.group(1).replace(',', ''))
                break

        # Special instructions
        notes_patterns = [
            r'Special\s*(?:Notes?|Instructions?):\s*(.+?)(?:\n|$)',
            r'Notes?:\s*(.+?)(?:\n|$)',
        ]
        for pattern in notes_patterns:
            match = re.search(pattern, markdown, re.IGNORECASE)
            if match:
                fields['special_instructions'] = match.group(1).strip()
                break

        return fields

    def _extract_tables(self, document) -> List[Dict]:
        """Extract tables from Docling document."""
        tables = []

        # Iterate through document elements
        for element in document.iterate_items():
            if hasattr(element, 'label') and 'table' in str(element.label).lower():
                # Extract table data
                table_data = {
                    'rows': [],
                    'headers': [],
                }

                if hasattr(element, 'text'):
                    # Parse table text
                    lines = element.text.split('\n')
                    for line in lines:
                        cells = [c.strip() for c in line.split('|') if c.strip()]
                        if cells:
                            if not table_data['headers']:
                                table_data['headers'] = cells
                            else:
                                table_data['rows'].append(cells)

                tables.append(table_data)

        return tables

    def _normalize_date(self, date_str: str) -> str:
        """Convert date to YYYY-MM-DD format."""
        from dateutil import parser
        try:
            parsed = parser.parse(date_str)
            return parsed.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            return date_str

    def _calculate_confidence(
        self,
        fields: Dict[str, str],
        tables: List[Dict],
    ) -> float:
        """
        Calculate confidence score based on extraction quality.

        Factors:
        - Number of required fields extracted
        - Table structure detected
        - Field format validation
        """
        score = 0.0
        total_weight = 0.0

        # Required fields (weighted)
        required_fields = {
            'order_id': 0.2,
            'client_name': 0.15,
            'order_date': 0.15,
            'delivery_date': 0.1,
            'order_total': 0.2,
        }

        for field, weight in required_fields.items():
            total_weight += weight
            if field in fields and fields[field]:
                score += weight

        # Table extraction bonus
        if tables:
            score += 0.1
            total_weight += 0.1

        # Special instructions bonus
        if 'special_instructions' in fields:
            score += 0.1
            total_weight += 0.1

        return score / total_weight if total_weight > 0 else 0.0

    def extract_to_schema(
        self,
        file_path: Union[str, Path],
        target_schema: Dict,
    ) -> Dict:
        """
        Extract and map to target JSON schema.

        Args:
            file_path: Document path
            target_schema: Expected output schema

        Returns:
            Dict matching target schema structure
        """
        result = self.extract(file_path)

        # Map extracted fields to schema
        output = {}

        for field_name, field_info in target_schema.items():
            if field_name in result.fields:
                output[field_name] = result.fields[field_name]
            else:
                # Use default value or None
                output[field_name] = field_info.get('default', None)

        return output


def extract_with_docling(file_path: str, verbose: bool = True) -> Dict:
    """
    Convenience function to extract document with Docling.

    Args:
        file_path: Path to document
        verbose: Print progress

    Returns:
        Extraction result as dict
    """
    if verbose:
        print(f"[Docling AI] Processing: {Path(file_path).name}")

    extractor = DoclingExtractor()
    result = extractor.extract(file_path)

    if verbose:
        print(f"[Docling AI] Extracted {len(result.fields)} fields")
        print(f"[Docling AI] Processing time: {result.processing_time:.2f}s")
        print(f"[Docling AI] Confidence: {result.confidence:.2f}")

    return {
        'fields': result.fields,
        'tables': result.tables,
        'markdown': result.markdown,
        'confidence': result.confidence,
        'method': result.extraction_method,
    }
