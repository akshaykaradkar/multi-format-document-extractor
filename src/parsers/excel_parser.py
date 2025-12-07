"""
Excel Parser for multi-sheet workbooks.

Handles Client B (Global Manufacturing Inc) - Multi-sheet Excel workbook
with custom field names and data spread across sheets.
"""

import re
from pathlib import Path
from typing import Union

import pandas as pd

from .base_parser import BaseParser, ParseError
from ..schemas import RawExtraction


class ExcelParser(BaseParser):
    """
    Parser for Excel workbooks with data across multiple sheets.

    Handles custom field naming conventions and consolidates
    data from multiple sheets into a single order.
    """

    @property
    def supported_extensions(self) -> list[str]:
        return [".xlsx", ".xls"]

    def parse(self) -> RawExtraction:
        """
        Parse Excel workbook and extract order data.

        Returns:
            RawExtraction with parsed order data
        """
        try:
            # Read all sheets
            xlsx = pd.ExcelFile(self.file_path)
            sheets = {name: pd.read_excel(xlsx, sheet_name=name) for name in xlsx.sheet_names}

            return self._extract_order_data(sheets)

        except Exception as e:
            raise ParseError(
                f"Failed to parse Excel: {str(e)}",
                file_path=self.file_path,
                original_error=e,
            )

    def _extract_order_data(self, sheets: dict[str, pd.DataFrame]) -> RawExtraction:
        """
        Extract order data from multiple sheets.

        Expected sheet structure:
        - Order_Info: Order#, Client_Name, Order_Created, Needed_By
        - Line_Items: SKU, Item_Desc, Order_Qty, Price_Each
        - Notes: Special_Requirements, Delivery_Instructions

        Args:
            sheets: Dict of sheet name to DataFrame

        Returns:
            RawExtraction object
        """
        order_info = self._find_sheet(sheets, ["Order_Info", "Order", "Info", "Header"])
        line_items = self._find_sheet(sheets, ["Line_Items", "Items", "Products", "Details"])
        notes = self._find_sheet(sheets, ["Notes", "Instructions", "Comments"])

        # Extract from order info sheet
        order_id = None
        client_name = None
        order_date = None
        delivery_date = None

        if order_info is not None:
            order_id = self._get_field_value(
                order_info, ["Order#", "Order_Number", "OrderNumber", "PO", "PO_Number", "Order"]
            )
            client_name = self._get_field_value(
                order_info, ["Client_Name", "ClientName", "Client", "Customer", "Company"]
            )
            order_date = self._get_field_value(
                order_info, ["Order_Created", "OrderDate", "Order_Date", "Date", "Created"]
            )
            delivery_date = self._get_field_value(
                order_info, ["Needed_By", "NeededBy", "Delivery_Date", "DeliveryDate", "Ship_Date", "Due_Date"]
            )

        # Normalize dates
        if order_date:
            order_date = self._normalize_date(order_date)
        if delivery_date:
            delivery_date = self._normalize_date(delivery_date)

        # Extract line items
        items = []
        if line_items is not None:
            items = self._extract_items(line_items)

        # Extract special instructions
        special_instructions = None
        if notes is not None:
            special_instructions = self._extract_notes(notes)

        # Calculate total from items
        order_total = sum(item.get("total_price", 0) for item in items)

        return RawExtraction(
            order_id=str(order_id) if order_id else None,
            client_name=str(client_name) if client_name else None,
            order_date=order_date,
            delivery_date=delivery_date,
            items=items,
            order_total=order_total,
            currency="USD",
            special_instructions=special_instructions,
            source_confidence=0.92,  # High confidence for structured Excel
            extraction_method="excel_pandas",
        )

    def _find_sheet(
        self, sheets: dict[str, pd.DataFrame], possible_names: list[str]
    ) -> pd.DataFrame | None:
        """Find a sheet by possible names (case-insensitive)."""
        sheet_names_lower = {name.lower(): name for name in sheets.keys()}

        for possible in possible_names:
            possible_lower = possible.lower()
            if possible_lower in sheet_names_lower:
                return sheets[sheet_names_lower[possible_lower]]

            # Try partial match
            for name_lower, name in sheet_names_lower.items():
                if possible_lower in name_lower or name_lower in possible_lower:
                    return sheets[name]

        # Return first sheet if only one exists
        if len(sheets) == 1:
            return list(sheets.values())[0]

        return None

    def _get_field_value(
        self, df: pd.DataFrame, possible_columns: list[str]
    ) -> str | None:
        """Get value from first matching column."""
        # Try exact column match (case-insensitive)
        col_map = {col.lower(): col for col in df.columns}

        for possible in possible_columns:
            possible_lower = possible.lower()
            if possible_lower in col_map:
                actual_col = col_map[possible_lower]
                value = df[actual_col].iloc[0] if len(df) > 0 else None
                if pd.notna(value):
                    return str(value)

        # Try partial match
        for possible in possible_columns:
            for col_lower, col in col_map.items():
                if possible.lower() in col_lower:
                    value = df[col].iloc[0] if len(df) > 0 else None
                    if pd.notna(value):
                        return str(value)

        # Try looking in first row/column for key-value pairs
        if len(df.columns) >= 2 and len(df) > 0:
            for idx, row in df.iterrows():
                key = str(row.iloc[0]).lower() if pd.notna(row.iloc[0]) else ""
                for possible in possible_columns:
                    if possible.lower() in key:
                        value = row.iloc[1] if pd.notna(row.iloc[1]) else None
                        if value:
                            return str(value)

        return None

    def _extract_items(self, df: pd.DataFrame) -> list[dict]:
        """Extract line items from items DataFrame."""
        items = []

        # Map columns
        col_map = self._map_columns(df.columns)

        for _, row in df.iterrows():
            item = self._parse_row(row, col_map)
            if item and item.get("description"):
                items.append(item)

        return items

    def _map_columns(self, columns: pd.Index) -> dict[str, str]:
        """Map expected fields to actual column names."""
        col_map = {}
        column_lower = {str(c).lower(): str(c) for c in columns}

        mappings = {
            "product_code": ["sku", "item_sku", "code", "product_code", "itemcode", "item"],
            "description": ["item_desc", "description", "desc", "product", "name", "item_name"],
            "quantity": ["order_qty", "qty", "quantity", "amount", "units"],
            "unit_price": ["price_each", "unit_price", "price", "rate", "cost"],
            "total_price": ["total", "line_total", "amount", "ext_price"],
        }

        for field, possible in mappings.items():
            for p in possible:
                p_lower = p.lower()
                if p_lower in column_lower:
                    col_map[field] = column_lower[p_lower]
                    break
                # Partial match
                for col_l, col in column_lower.items():
                    if p_lower in col_l:
                        col_map[field] = col
                        break
                if field in col_map:
                    break

        return col_map

    def _parse_row(self, row: pd.Series, col_map: dict[str, str]) -> dict | None:
        """Parse a single row into an item dict."""
        try:
            item = {}

            # Get description (required)
            if "description" in col_map:
                desc = row.get(col_map["description"])
                if pd.notna(desc) and str(desc).strip():
                    item["description"] = str(desc).strip()
                else:
                    return None
            else:
                return None

            # Get product code
            if "product_code" in col_map:
                code = row.get(col_map["product_code"])
                if pd.notna(code):
                    item["product_code"] = str(code).strip()
                else:
                    item["product_code"] = self._generate_code(item["description"])
            else:
                item["product_code"] = self._generate_code(item["description"])

            # Get quantity
            if "quantity" in col_map:
                qty = row.get(col_map["quantity"])
                item["quantity"] = int(float(qty)) if pd.notna(qty) else 1
            else:
                item["quantity"] = 1

            # Get unit price
            if "unit_price" in col_map:
                price = row.get(col_map["unit_price"])
                item["unit_price"] = float(price) if pd.notna(price) else 0.0
            else:
                item["unit_price"] = 0.0

            # Get or calculate total
            if "total_price" in col_map:
                total = row.get(col_map["total_price"])
                item["total_price"] = float(total) if pd.notna(total) else 0.0
            else:
                item["total_price"] = item["quantity"] * item["unit_price"]

            return item

        except (ValueError, TypeError):
            return None

    def _extract_notes(self, df: pd.DataFrame) -> str | None:
        """Extract special instructions from notes sheet."""
        notes = []

        # Try common column names
        possible_cols = [
            "Special_Requirements",
            "Delivery_Instructions",
            "Notes",
            "Comments",
            "Instructions",
        ]

        col_lower = {str(c).lower(): str(c) for c in df.columns}

        for possible in possible_cols:
            p_lower = possible.lower()
            for col_l, col in col_lower.items():
                if p_lower in col_l or col_l in p_lower:
                    for _, row in df.iterrows():
                        value = row.get(col)
                        if pd.notna(value) and str(value).strip():
                            notes.append(str(value).strip())

        # Also check if it's a simple list
        if not notes and len(df.columns) >= 1:
            for _, row in df.iterrows():
                for val in row:
                    if pd.notna(val) and str(val).strip():
                        notes.append(str(val).strip())

        return "; ".join(notes) if notes else None

    def _normalize_date(self, date_value) -> str | None:
        """Normalize date to YYYY-MM-DD format."""
        if pd.isna(date_value):
            return None

        # Handle pandas Timestamp
        if hasattr(date_value, "strftime"):
            return date_value.strftime("%Y-%m-%d")

        # Parse string dates
        from dateutil import parser

        try:
            parsed = parser.parse(str(date_value))
            return parsed.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            return str(date_value)

    def _generate_code(self, description: str) -> str:
        """Generate product code from description."""
        if not description:
            return "ITEM"
        words = description.split()[:3]
        return "".join(w[0].upper() for w in words if w) or "ITEM"
