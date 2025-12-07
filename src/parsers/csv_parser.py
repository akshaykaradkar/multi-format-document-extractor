"""
CSV Parser for structured data files.

Handles Client D (Supply Chain Partners) - CSV with varying column
structures, different column orders, and mixed date formats.
"""

import re
from pathlib import Path
from typing import Union

import pandas as pd

from .base_parser import BaseParser, ParseError
from ..schemas import RawExtraction


class CSVParser(BaseParser):
    """
    Parser for CSV files with varying structures.

    Handles different column orderings, mixed date formats,
    and multi-row orders (grouped by order ID).
    """

    @property
    def supported_extensions(self) -> list[str]:
        return [".csv"]

    def parse(self) -> RawExtraction:
        """
        Parse CSV file and extract order data.

        Returns:
            RawExtraction with parsed order data
        """
        try:
            # Try different encodings
            df = None
            for encoding in ["utf-8", "latin-1", "cp1252"]:
                try:
                    df = pd.read_csv(self.file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if df is None:
                raise ParseError("Could not decode CSV file with known encodings")

            return self._extract_order_data(df)

        except Exception as e:
            raise ParseError(
                f"Failed to parse CSV: {str(e)}",
                file_path=self.file_path,
                original_error=e,
            )

    def _extract_order_data(self, df: pd.DataFrame) -> RawExtraction:
        """
        Extract order data from CSV DataFrame.

        Args:
            df: Pandas DataFrame from CSV

        Returns:
            RawExtraction object
        """
        # Map columns to standard names
        col_map = self._map_columns(df.columns)

        # Check if this is a multi-row order (grouped by order ID)
        order_id_col = col_map.get("order_id")
        if order_id_col and df[order_id_col].nunique() == 1:
            # All rows are same order - process as single order
            return self._process_single_order(df, col_map)
        elif order_id_col:
            # Multiple orders - take the first one for PoC
            first_order_id = df[order_id_col].iloc[0]
            order_df = df[df[order_id_col] == first_order_id]
            return self._process_single_order(order_df, col_map)
        else:
            # No order ID column - treat entire file as one order
            return self._process_single_order(df, col_map)

    def _map_columns(self, columns: pd.Index) -> dict[str, str]:
        """Map CSV columns to standard field names."""
        col_map = {}
        col_lower = {str(c).lower().strip(): str(c) for c in columns}

        # Define possible column name variations
        mappings = {
            "order_id": [
                "order_id",
                "orderid",
                "order_number",
                "ordernumber",
                "po",
                "po_number",
                "order#",
                "order",
            ],
            "client_name": [
                "customer",
                "client",
                "client_name",
                "company",
                "customer_name",
                "buyer",
            ],
            "product_code": [
                "item_sku",
                "sku",
                "product_code",
                "item_code",
                "code",
                "product_id",
            ],
            "description": [
                "product_name",
                "item_name",
                "description",
                "item_desc",
                "product",
                "item",
                "name",
            ],
            "quantity": [
                "qty_ordered",
                "qty",
                "quantity",
                "order_qty",
                "amount",
                "units",
            ],
            "unit_price": [
                "unit_cost",
                "price",
                "unit_price",
                "cost",
                "price_each",
                "rate",
            ],
            "order_date": [
                "order_date",
                "orderdate",
                "date",
                "created",
                "order_created",
            ],
            "delivery_date": [
                "ship_date",
                "shipdate",
                "delivery_date",
                "needed_by",
                "due_date",
            ],
            "notes": ["notes", "comments", "instructions", "special_instructions"],
        }

        for field, variations in mappings.items():
            for var in variations:
                var_lower = var.lower()
                if var_lower in col_lower:
                    col_map[field] = col_lower[var_lower]
                    break

        return col_map

    def _process_single_order(
        self, df: pd.DataFrame, col_map: dict[str, str]
    ) -> RawExtraction:
        """Process a single order from DataFrame rows."""
        # Get order-level info from first row
        first_row = df.iloc[0]

        # Order ID
        order_id = None
        if "order_id" in col_map:
            val = first_row[col_map["order_id"]]
            order_id = str(val) if pd.notna(val) else None

        # Client name
        client_name = None
        if "client_name" in col_map:
            val = first_row[col_map["client_name"]]
            client_name = str(val) if pd.notna(val) else None

        # Order date
        order_date = None
        if "order_date" in col_map:
            val = first_row[col_map["order_date"]]
            order_date = self._normalize_date(val)

        # Delivery date
        delivery_date = None
        if "delivery_date" in col_map:
            val = first_row[col_map["delivery_date"]]
            delivery_date = self._normalize_date(val)

        # Extract line items from all rows
        items = self._extract_items(df, col_map)

        # Calculate total
        order_total = sum(item.get("total_price", 0) for item in items)

        # Notes
        special_instructions = None
        if "notes" in col_map:
            notes = df[col_map["notes"]].dropna().unique()
            notes = [str(n).strip() for n in notes if str(n).strip()]
            special_instructions = "; ".join(notes) if notes else None

        return RawExtraction(
            order_id=order_id,
            client_name=client_name,
            order_date=order_date,
            delivery_date=delivery_date,
            items=items,
            order_total=order_total,
            currency="USD",
            special_instructions=special_instructions,
            source_confidence=0.95,  # High confidence for structured CSV
            extraction_method="csv_pandas",
        )

    def _extract_items(self, df: pd.DataFrame, col_map: dict[str, str]) -> list[dict]:
        """Extract line items from DataFrame rows."""
        items = []

        for _, row in df.iterrows():
            item = {}

            # Description (required)
            if "description" in col_map:
                desc = row.get(col_map["description"])
                if pd.notna(desc) and str(desc).strip():
                    item["description"] = str(desc).strip()
                else:
                    continue
            else:
                continue

            # Product code
            if "product_code" in col_map:
                code = row.get(col_map["product_code"])
                if pd.notna(code):
                    item["product_code"] = str(code).strip()
                else:
                    item["product_code"] = self._generate_code(item["description"])
            else:
                item["product_code"] = self._generate_code(item["description"])

            # Quantity
            if "quantity" in col_map:
                qty = row.get(col_map["quantity"])
                try:
                    item["quantity"] = int(float(qty)) if pd.notna(qty) else 1
                except (ValueError, TypeError):
                    item["quantity"] = 1
            else:
                item["quantity"] = 1

            # Unit price
            if "unit_price" in col_map:
                price = row.get(col_map["unit_price"])
                try:
                    item["unit_price"] = float(price) if pd.notna(price) else 0.0
                except (ValueError, TypeError):
                    item["unit_price"] = 0.0
            else:
                item["unit_price"] = 0.0

            # Calculate total
            item["total_price"] = item["quantity"] * item["unit_price"]

            items.append(item)

        return items

    def _normalize_date(self, date_value) -> str | None:
        """Normalize date to YYYY-MM-DD format."""
        if pd.isna(date_value):
            return None

        # Handle pandas Timestamp
        if hasattr(date_value, "strftime"):
            return date_value.strftime("%Y-%m-%d")

        date_str = str(date_value).strip()
        if not date_str:
            return None

        # Try various date formats
        from dateutil import parser

        try:
            # Handle formats like "03/27/2024" or "2024-03-27"
            parsed = parser.parse(date_str, dayfirst=False)
            return parsed.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            # If parsing fails, return as-is if it looks like a date
            if re.match(r"\d{4}-\d{2}-\d{2}", date_str):
                return date_str
            return None

    def _generate_code(self, description: str) -> str:
        """Generate product code from description."""
        if not description:
            return "ITEM"
        words = description.split()[:3]
        return "".join(w[0].upper() for w in words if w and w[0].isalnum()) or "ITEM"
