"""
PDF Parser for structured PDF invoices.

Handles Client A (TechCorp Solutions) - Clean PDF with tabular layout.
Uses pdfplumber for table and text extraction.
"""

import re
from pathlib import Path
from typing import Union

import pdfplumber

from .base_parser import BaseParser, ParseError
from ..schemas import RawExtraction


class PDFParser(BaseParser):
    """
    Parser for PDF documents with tabular layouts.

    Optimized for structured invoices with clear field labels
    and table-formatted line items.
    """

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def parse(self) -> RawExtraction:
        """
        Parse PDF document and extract order data.

        Returns:
            RawExtraction with parsed order data
        """
        try:
            with pdfplumber.open(self.file_path) as pdf:
                # Extract all text and tables from all pages
                all_text = ""
                all_tables = []

                for page in pdf.pages:
                    text = page.extract_text() or ""
                    all_text += text + "\n"

                    tables = page.extract_tables()
                    if tables:
                        all_tables.extend(tables)

                # Parse the extracted content
                return self._extract_order_data(all_text, all_tables)

        except Exception as e:
            raise ParseError(
                f"Failed to parse PDF: {str(e)}",
                file_path=self.file_path,
                original_error=e,
            )

    def _extract_order_data(
        self, text: str, tables: list[list[list[str]]]
    ) -> RawExtraction:
        """
        Extract order data from PDF text and tables.

        Args:
            text: Extracted text from PDF
            tables: List of tables extracted from PDF

        Returns:
            RawExtraction object
        """
        # Extract order ID
        order_id = self._extract_pattern(
            text,
            [
                r"(?:PURCHASE ORDER|PO|Order)\s*#?\s*:?\s*([A-Z0-9\-]+)",
                r"Order\s*(?:Number|No\.?|#)\s*:?\s*([A-Z0-9\-]+)",
            ],
        )

        # Extract client name - look for company name patterns
        client_name = self._extract_client_name(text)

        # Extract dates
        order_date = self._extract_date(text, ["Date:", "Order Date:", "Dated:"])
        delivery_date = self._extract_date(
            text, ["Delivery Required:", "Need by:", "Delivery Date:", "Ship Date:"]
        )

        # Extract line items from tables
        items = self._extract_items_from_tables(tables, text)

        # Extract total
        order_total = self._extract_total(text)

        # Extract special instructions
        special_instructions = self._extract_special_instructions(text)

        return RawExtraction(
            order_id=order_id,
            client_name=client_name,
            order_date=order_date,
            delivery_date=delivery_date,
            items=items,
            order_total=order_total,
            currency="USD",
            special_instructions=special_instructions,
            source_confidence=0.95,  # High confidence for structured PDFs
            extraction_method="pdf_pdfplumber",
        )

    def _extract_pattern(self, text: str, patterns: list[str]) -> str | None:
        """Try multiple regex patterns and return first match."""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _extract_client_name(self, text: str) -> str | None:
        """Extract client/company name from text."""
        # Look for common patterns
        patterns = [
            r"^([A-Z][A-Za-z\s]+(?:Solutions|Inc|Corp|LLC|Ltd|Co\.?))\s*$",
            r"From:\s*([A-Za-z\s]+(?:Solutions|Inc|Corp|LLC|Ltd|Co\.?))",
            r"Client:\s*([A-Za-z\s]+)",
            r"Company:\s*([A-Za-z\s]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Fallback: look for lines with company-like names
        lines = text.split("\n")
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if any(
                suffix in line
                for suffix in ["Solutions", "Inc", "Corp", "LLC", "Ltd", "Co."]
            ):
                return line

        return None

    def _extract_date(self, text: str, labels: list[str]) -> str | None:
        """Extract and normalize date following given labels."""
        for label in labels:
            # Pattern for label followed by date
            pattern = rf"{re.escape(label)}\s*([A-Za-z]+\s+\d{{1,2}},?\s+\d{{4}}|\d{{1,2}}[/\-]\d{{1,2}}[/\-]\d{{2,4}})"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self._normalize_date(match.group(1))

        return None

    def _normalize_date(self, date_str: str) -> str:
        """Convert various date formats to YYYY-MM-DD."""
        from dateutil import parser

        try:
            parsed = parser.parse(date_str)
            return parsed.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            return date_str

    def _extract_items_from_tables(
        self, tables: list[list[list[str]]], text: str
    ) -> list[dict]:
        """Extract line items from tables or text."""
        items = []

        for table in tables:
            if not table or len(table) < 2:
                continue

            # Try to identify header row
            header_row = None
            data_start = 0

            for i, row in enumerate(table):
                row_text = " ".join(str(cell or "").lower() for cell in row)
                if any(
                    keyword in row_text
                    for keyword in ["item", "description", "qty", "quantity", "price"]
                ):
                    header_row = row
                    data_start = i + 1
                    break

            if header_row is None:
                # Assume first row is header
                header_row = table[0]
                data_start = 1

            # Map columns
            col_map = self._map_columns(header_row)

            # Extract data rows
            for row in table[data_start:]:
                item = self._parse_table_row(row, col_map)
                if item and item.get("description"):
                    items.append(item)

        # If no items from tables, try to parse from text
        if not items:
            items = self._extract_items_from_text(text)

        return items

    def _map_columns(self, header: list[str]) -> dict[str, int]:
        """Map column names to indices."""
        col_map = {}
        keywords = {
            "code": ["code", "sku", "item code", "product code"],
            "description": ["description", "desc", "item", "product", "name"],
            "quantity": ["qty", "quantity", "amount", "units"],
            "unit_price": ["unit price", "price", "rate", "unit cost"],
            "total": ["total", "amount", "line total", "ext"],
        }

        for i, cell in enumerate(header):
            cell_lower = str(cell or "").lower().strip()
            for key, variations in keywords.items():
                if any(v in cell_lower for v in variations):
                    col_map[key] = i
                    break

        return col_map

    def _parse_table_row(self, row: list[str], col_map: dict[str, int]) -> dict | None:
        """Parse a single table row into an item dict."""
        try:
            item = {}

            # Get description (required)
            desc_idx = col_map.get("description")
            if desc_idx is not None and desc_idx < len(row):
                item["description"] = str(row[desc_idx] or "").strip()
            else:
                return None

            # Get product code
            code_idx = col_map.get("code")
            if code_idx is not None and code_idx < len(row):
                item["product_code"] = str(row[code_idx] or "").strip()
            else:
                # Generate code from description
                item["product_code"] = self._generate_product_code(item["description"])

            # Get quantity
            qty_idx = col_map.get("quantity")
            if qty_idx is not None and qty_idx < len(row):
                item["quantity"] = self._parse_number(row[qty_idx], as_int=True)
            else:
                item["quantity"] = 1

            # Get unit price
            price_idx = col_map.get("unit_price")
            if price_idx is not None and price_idx < len(row):
                item["unit_price"] = self._parse_number(row[price_idx])
            else:
                item["unit_price"] = 0.0

            # Get or calculate total
            total_idx = col_map.get("total")
            if total_idx is not None and total_idx < len(row):
                item["total_price"] = self._parse_number(row[total_idx])
            else:
                item["total_price"] = item["quantity"] * item["unit_price"]

            return item if item.get("description") else None

        except (ValueError, IndexError):
            return None

    def _extract_items_from_text(self, text: str) -> list[dict]:
        """Extract items from unstructured text."""
        items = []
        # Pattern for item lines like "TC-001 | Widget Pro | 50 | $25.00 | $1,250.00"
        pattern = r"([A-Z0-9\-]+)\s*\|\s*([^|]+)\s*\|\s*(\d+)\s*\|\s*\$?([\d,\.]+)\s*\|\s*\$?([\d,\.]+)"

        for match in re.finditer(pattern, text):
            items.append(
                {
                    "product_code": match.group(1).strip(),
                    "description": match.group(2).strip(),
                    "quantity": int(match.group(3)),
                    "unit_price": float(match.group(4).replace(",", "")),
                    "total_price": float(match.group(5).replace(",", "")),
                }
            )

        return items

    def _extract_total(self, text: str) -> float | None:
        """Extract order total from text."""
        patterns = [
            r"TOTAL:?\s*\$?([\d,]+\.?\d*)",
            r"Grand Total:?\s*\$?([\d,]+\.?\d*)",
            r"Order Total:?\s*\$?([\d,]+\.?\d*)",
            r"Amount Due:?\s*\$?([\d,]+\.?\d*)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1).replace(",", ""))

        return None

    def _extract_special_instructions(self, text: str) -> str | None:
        """Extract special instructions or notes."""
        patterns = [
            r"Special (?:Notes?|Instructions?):?\s*(.+?)(?:\n|$)",
            r"Notes?:?\s*(.+?)(?:\n|$)",
            r"Instructions?:?\s*(.+?)(?:\n|$)",
            r"Comments?:?\s*(.+?)(?:\n|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                instruction = match.group(1).strip()
                if instruction and len(instruction) > 3:
                    return instruction

        return None

    def _parse_number(self, value: str, as_int: bool = False) -> float | int:
        """Parse a number from string, handling currency symbols and commas."""
        if value is None:
            return 0 if as_int else 0.0

        # Remove currency symbols and whitespace
        cleaned = re.sub(r"[^\d.\-]", "", str(value))

        try:
            if as_int:
                return int(float(cleaned))
            return float(cleaned)
        except ValueError:
            return 0 if as_int else 0.0

    def _generate_product_code(self, description: str) -> str:
        """Generate a product code from description."""
        if not description:
            return "UNKNOWN"

        # Take first letters of words, uppercase
        words = description.split()[:3]
        code = "".join(w[0].upper() for w in words if w)
        return code or "ITEM"
