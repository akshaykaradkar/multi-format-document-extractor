"""
Confidence Scorer for extraction quality assessment.

Calculates a confidence score (0.0-1.0) based on:
- Field completeness
- Source extraction confidence
- Schema validation results
"""

from typing import Optional

from ..schemas import RawExtraction, StandardizedOrder
from ..config import CONFIDENCE_AUTO_APPROVE, CONFIDENCE_REVIEW_THRESHOLD


class ConfidenceScorer:
    """
    Calculates confidence scores for extracted order data.

    Score Components (weighted):
    - Field Completeness: 40% - Are all required fields populated?
    - Source Confidence: 40% - How confident was the parser/OCR?
    - Validation Score: 20% - Does data pass validation checks?
    """

    # Weight configuration
    WEIGHT_COMPLETENESS = 0.40
    WEIGHT_SOURCE = 0.40
    WEIGHT_VALIDATION = 0.20

    # Required fields for completeness calculation
    REQUIRED_FIELDS = [
        "order_id",
        "client_name",
        "order_date",
        "delivery_date",
        "items",
        "order_total",
    ]

    def calculate_score(
        self,
        raw: RawExtraction,
        validated: bool = True,
        validation_errors: list[str] = None,
    ) -> float:
        """
        Calculate composite confidence score.

        Args:
            raw: RawExtraction from parser
            validated: Whether the data passed schema validation
            validation_errors: List of validation error messages

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Calculate individual components
        completeness = self._calculate_completeness(raw)
        source_confidence = raw.source_confidence
        validation = self._calculate_validation_score(validated, validation_errors)

        # Weighted average
        score = (
            completeness * self.WEIGHT_COMPLETENESS
            + source_confidence * self.WEIGHT_SOURCE
            + validation * self.WEIGHT_VALIDATION
        )

        return round(min(1.0, max(0.0, score)), 2)

    def _calculate_completeness(self, raw: RawExtraction) -> float:
        """
        Calculate field completeness score.

        Returns percentage of required fields that are populated.
        """
        populated = 0
        total = len(self.REQUIRED_FIELDS)

        for field in self.REQUIRED_FIELDS:
            value = getattr(raw, field, None)
            if value is not None:
                # Special handling for lists
                if isinstance(value, list):
                    if len(value) > 0:
                        populated += 1
                # Special handling for strings
                elif isinstance(value, str):
                    if value.strip():
                        populated += 1
                else:
                    populated += 1

        # Bonus for having multiple items (shows comprehensive extraction)
        if raw.items and len(raw.items) > 1:
            populated += 0.5

        return min(1.0, populated / total)

    def _calculate_validation_score(
        self, validated: bool, errors: list[str] = None
    ) -> float:
        """
        Calculate validation score based on schema compliance.
        """
        if validated and (not errors or len(errors) == 0):
            return 1.0

        if errors:
            # Reduce score based on number of errors
            error_penalty = min(0.8, len(errors) * 0.15)
            return max(0.2, 1.0 - error_penalty)

        return 0.5 if validated else 0.3

    def get_confidence_status(self, score: float) -> dict:
        """
        Get status and recommendation based on confidence score.

        Args:
            score: Confidence score (0.0-1.0)

        Returns:
            Dict with status, recommendation, and thresholds
        """
        if score >= CONFIDENCE_AUTO_APPROVE:
            return {
                "status": "HIGH",
                "color": "green",
                "recommendation": "Auto-approve for processing",
                "action": "APPROVE",
                "requires_review": False,
            }
        elif score >= CONFIDENCE_REVIEW_THRESHOLD:
            return {
                "status": "MEDIUM",
                "color": "yellow",
                "recommendation": "Review recommended before processing",
                "action": "REVIEW",
                "requires_review": True,
            }
        else:
            return {
                "status": "LOW",
                "color": "red",
                "recommendation": "Manual review required",
                "action": "MANUAL_REVIEW",
                "requires_review": True,
            }

    def generate_report(self, raw: RawExtraction, score: float) -> str:
        """
        Generate a human-readable confidence report.

        Args:
            raw: RawExtraction data
            score: Calculated confidence score

        Returns:
            Formatted report string
        """
        status = self.get_confidence_status(score)
        completeness = self._calculate_completeness(raw)

        # Identify missing fields
        missing = []
        for field in self.REQUIRED_FIELDS:
            value = getattr(raw, field, None)
            if value is None or (isinstance(value, list) and len(value) == 0):
                missing.append(field)
            elif isinstance(value, str) and not value.strip():
                missing.append(field)

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║              CONFIDENCE SCORE REPORT                          ║
╠══════════════════════════════════════════════════════════════╣
║ Overall Score: {score:.2f} ({status['status']})
║ Recommendation: {status['recommendation']}
╠══════════════════════════════════════════════════════════════╣
║ Score Breakdown:
║   • Field Completeness: {completeness:.0%} (weight: {self.WEIGHT_COMPLETENESS:.0%})
║   • Source Confidence:  {raw.source_confidence:.0%} (weight: {self.WEIGHT_SOURCE:.0%})
║   • Extraction Method:  {raw.extraction_method}
╠══════════════════════════════════════════════════════════════╣
║ Extracted Fields:
║   • Order ID:      {'✓' if raw.order_id else '✗'} {raw.order_id or 'Missing'}
║   • Client Name:   {'✓' if raw.client_name else '✗'} {raw.client_name or 'Missing'}
║   • Order Date:    {'✓' if raw.order_date else '✗'} {raw.order_date or 'Missing'}
║   • Delivery Date: {'✓' if raw.delivery_date else '✗'} {raw.delivery_date or 'Missing'}
║   • Items:         {'✓' if raw.items else '✗'} {len(raw.items)} item(s)
║   • Order Total:   {'✓' if raw.order_total else '✗'} ${raw.order_total or 0:.2f}
"""

        if missing:
            report += f"""╠══════════════════════════════════════════════════════════════╣
║ ⚠ Missing Fields: {', '.join(missing)}
"""

        report += """╚══════════════════════════════════════════════════════════════╝"""

        return report
