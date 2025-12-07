"""
Schema Validator for standardized order validation.

Validates transformed data against the target JSON schema
using Pydantic models.
"""

from typing import Tuple, List
from pydantic import ValidationError

from ..schemas import StandardizedOrder, OrderItem


class SchemaValidator:
    """
    Validates order data against the standardized schema.

    Uses Pydantic for validation with detailed error reporting.
    """

    def validate(self, data: dict) -> Tuple[bool, List[str]]:
        """
        Validate data dict against StandardizedOrder schema.

        Args:
            data: Dictionary of order data

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        try:
            StandardizedOrder(**data)
            return True, []
        except ValidationError as e:
            errors = self._format_errors(e)
            return False, errors

    def validate_order(self, order: StandardizedOrder) -> Tuple[bool, List[str]]:
        """
        Validate an already-constructed StandardizedOrder.

        Args:
            order: StandardizedOrder instance

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Additional business logic validation
        errors.extend(self._validate_dates(order))
        errors.extend(self._validate_items(order))
        errors.extend(self._validate_totals(order))

        return len(errors) == 0, errors

    def _format_errors(self, validation_error: ValidationError) -> List[str]:
        """Format Pydantic validation errors into readable messages."""
        errors = []
        for error in validation_error.errors():
            loc = " -> ".join(str(l) for l in error["loc"])
            msg = error["msg"]
            errors.append(f"{loc}: {msg}")
        return errors

    def _validate_dates(self, order: StandardizedOrder) -> List[str]:
        """Validate date logic."""
        errors = []

        try:
            from datetime import datetime

            order_dt = datetime.strptime(order.order_date, "%Y-%m-%d")
            delivery_dt = datetime.strptime(order.delivery_date, "%Y-%m-%d")

            # Delivery should be on or after order date
            if delivery_dt < order_dt:
                errors.append(
                    f"Delivery date ({order.delivery_date}) is before order date ({order.order_date})"
                )

        except ValueError as e:
            errors.append(f"Date format error: {str(e)}")

        return errors

    def _validate_items(self, order: StandardizedOrder) -> List[str]:
        """Validate line items."""
        errors = []

        if not order.items:
            errors.append("Order must have at least one item")
            return errors

        for i, item in enumerate(order.items):
            # Check for reasonable values
            if item.quantity <= 0:
                errors.append(f"Item {i+1}: Quantity must be positive")

            if item.unit_price < 0:
                errors.append(f"Item {i+1}: Unit price cannot be negative")

            if item.total_price < 0:
                errors.append(f"Item {i+1}: Total price cannot be negative")

            # Check total calculation (allow small rounding difference)
            expected_total = item.quantity * item.unit_price
            if abs(item.total_price - expected_total) > 0.10:
                errors.append(
                    f"Item {i+1}: Total price ({item.total_price}) doesn't match quantity * unit_price ({expected_total:.2f})"
                )

        return errors

    def _validate_totals(self, order: StandardizedOrder) -> List[str]:
        """Validate order total matches sum of items."""
        errors = []

        calculated_total = sum(item.total_price for item in order.items)

        # Allow small rounding difference
        if abs(order.order_total - calculated_total) > 1.00:
            errors.append(
                f"Order total ({order.order_total}) doesn't match sum of items ({calculated_total:.2f})"
            )

        return errors

    def get_validation_summary(
        self, order: StandardizedOrder
    ) -> dict:
        """
        Get a summary of validation results.

        Args:
            order: StandardizedOrder to validate

        Returns:
            Dict with validation status and details
        """
        is_valid, errors = self.validate_order(order)

        return {
            "is_valid": is_valid,
            "error_count": len(errors),
            "errors": errors,
            "order_id": order.order_id,
            "item_count": len(order.items),
            "order_total": order.order_total,
            "confidence_score": order.confidence_score,
            "confidence_status": order.get_confidence_status(),
        }
