"""
Data Transformer for field mapping and normalization.

Transforms raw extracted data into the standardized schema format,
handling date normalization, field mapping, and data cleaning.
"""

import re
from datetime import datetime
from typing import Optional

from dateutil import parser as date_parser

from ..schemas import RawExtraction, StandardizedOrder, OrderItem
from ..config import DEFAULT_CURRENCY


class DataTransformer:
    """
    Transforms raw extracted data into standardized format.

    Handles:
    - Date normalization to YYYY-MM-DD
    - Field mapping from various naming conventions
    - Total calculation and validation
    - Currency detection
    - Missing field handling
    """

    def transform(
        self, raw: RawExtraction, confidence_score: float = 0.0
    ) -> StandardizedOrder:
        """
        Transform raw extraction to standardized order.

        Args:
            raw: RawExtraction from parser
            confidence_score: Pre-calculated confidence score

        Returns:
            StandardizedOrder conforming to target schema

        Raises:
            ValueError: If required fields cannot be transformed
        """
        # Transform items
        items = self._transform_items(raw.items)

        # Calculate order total if not provided or validate
        order_total = self._calculate_total(raw.order_total, items)

        # Normalize dates
        order_date = self._normalize_date(raw.order_date) or self._get_current_date()
        delivery_date = self._normalize_date(raw.delivery_date) or self._get_default_delivery_date(order_date)

        # Clean and validate required fields
        order_id = self._clean_string(raw.order_id) or self._generate_order_id()
        client_name = self._clean_string(raw.client_name) or "Unknown Client"

        # Detect currency
        currency = self._detect_currency(raw.currency)

        # Clean special instructions
        special_instructions = self._clean_string(raw.special_instructions)

        return StandardizedOrder(
            order_id=order_id,
            client_name=client_name,
            order_date=order_date,
            delivery_date=delivery_date,
            items=items,
            order_total=order_total,
            currency=currency,
            special_instructions=special_instructions,
            confidence_score=confidence_score,
        )

    def _transform_items(self, raw_items: list[dict]) -> list[OrderItem]:
        """Transform raw item dicts to OrderItem objects."""
        items = []

        for raw_item in raw_items:
            try:
                item = OrderItem(
                    product_code=self._clean_string(raw_item.get("product_code")) or "ITEM",
                    description=self._clean_string(raw_item.get("description")) or "Unknown Item",
                    quantity=self._safe_int(raw_item.get("quantity", 1)),
                    unit_price=self._safe_float(raw_item.get("unit_price", 0)),
                    total_price=self._safe_float(raw_item.get("total_price", 0)),
                )

                # Recalculate total if it seems wrong
                calculated_total = item.quantity * item.unit_price
                if abs(item.total_price - calculated_total) > 0.01 and calculated_total > 0:
                    item = OrderItem(
                        product_code=item.product_code,
                        description=item.description,
                        quantity=item.quantity,
                        unit_price=item.unit_price,
                        total_price=round(calculated_total, 2),
                    )

                items.append(item)

            except Exception:
                # Skip invalid items
                continue

        return items

    def _calculate_total(
        self, raw_total: Optional[float], items: list[OrderItem]
    ) -> float:
        """Calculate or validate order total."""
        calculated = sum(item.total_price for item in items)

        if raw_total is not None and raw_total > 0:
            # If difference is small, use raw total
            if abs(raw_total - calculated) < 1.0:
                return round(raw_total, 2)
            # If calculated seems more accurate, use that
            return round(calculated, 2)

        return round(calculated, 2)

    def _normalize_date(self, date_value: Optional[str]) -> Optional[str]:
        """Normalize date to YYYY-MM-DD format."""
        if not date_value:
            return None

        # Already in correct format
        if re.match(r"^\d{4}-\d{2}-\d{2}$", str(date_value)):
            return str(date_value)

        try:
            # Parse various date formats
            parsed = date_parser.parse(str(date_value), dayfirst=False)
            return parsed.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            return None

    def _get_current_date(self) -> str:
        """Get current date in YYYY-MM-DD format."""
        return datetime.now().strftime("%Y-%m-%d")

    def _get_default_delivery_date(self, order_date: str) -> str:
        """Get default delivery date (7 days from order)."""
        from datetime import timedelta

        try:
            order = datetime.strptime(order_date, "%Y-%m-%d")
            delivery = order + timedelta(days=7)
            return delivery.strftime("%Y-%m-%d")
        except ValueError:
            return self._get_current_date()

    def _clean_string(self, value: Optional[str]) -> Optional[str]:
        """Clean and normalize string value."""
        if value is None:
            return None

        # Convert to string and strip
        cleaned = str(value).strip()

        # Remove excessive whitespace
        cleaned = re.sub(r"\s+", " ", cleaned)

        # Return None for empty strings
        return cleaned if cleaned else None

    def _detect_currency(self, raw_currency: Optional[str]) -> str:
        """Detect currency from raw value or default."""
        if not raw_currency:
            return DEFAULT_CURRENCY

        currency = str(raw_currency).upper().strip()

        # Map common currency indicators
        currency_map = {
            "$": "USD",
            "USD": "USD",
            "US": "USD",
            "EUR": "EUR",
            "€": "EUR",
            "GBP": "GBP",
            "£": "GBP",
        }

        return currency_map.get(currency, DEFAULT_CURRENCY)

    def _generate_order_id(self) -> str:
        """Generate a unique order ID if none provided."""
        from datetime import datetime
        import random

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = random.randint(100, 999)
        return f"ORD-{timestamp}-{random_suffix}"

    def _safe_int(self, value) -> int:
        """Safely convert to integer."""
        try:
            if isinstance(value, str):
                cleaned = re.sub(r"[^\d]", "", value)
                return int(cleaned) if cleaned else 1
            return max(1, int(value))
        except (ValueError, TypeError):
            return 1

    def _safe_float(self, value) -> float:
        """Safely convert to float."""
        try:
            if isinstance(value, str):
                cleaned = re.sub(r"[^\d.]", "", value)
                return float(cleaned) if cleaned else 0.0
            return max(0.0, float(value))
        except (ValueError, TypeError):
            return 0.0
