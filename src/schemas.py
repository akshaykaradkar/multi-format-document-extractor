"""
Pydantic schemas for standardized order data.

Defines the target JSON schema that all parsed documents
must conform to.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class OrderItem(BaseModel):
    """Individual line item in an order."""

    product_code: str = Field(..., description="Product SKU or code")
    description: str = Field(..., description="Product description")
    quantity: int = Field(..., ge=1, description="Quantity ordered")
    unit_price: float = Field(..., ge=0, description="Price per unit")
    total_price: float = Field(..., ge=0, description="Line item total")

    @field_validator("product_code", "description")
    @classmethod
    def non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class StandardizedOrder(BaseModel):
    """
    Standardized order schema.

    All processed orders must conform to this structure.
    """

    order_id: str = Field(..., description="Unique order identifier")
    client_name: str = Field(..., description="Client/company name")
    order_date: str = Field(..., description="Order date in YYYY-MM-DD format")
    delivery_date: str = Field(..., description="Delivery date in YYYY-MM-DD format")
    items: list[OrderItem] = Field(..., min_length=1, description="Order line items")
    order_total: float = Field(..., ge=0, description="Total order amount")
    currency: str = Field(default="USD", description="Currency code")
    special_instructions: Optional[str] = Field(
        default=None, description="Special delivery or handling instructions"
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Extraction confidence (0.0-1.0)"
    )

    @field_validator("order_id", "client_name")
    @classmethod
    def non_empty_string(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()

    @field_validator("order_date", "delivery_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Ensure date is in YYYY-MM-DD format."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Date must be in YYYY-MM-DD format, got: {v}")
        return v

    def get_confidence_status(self) -> str:
        """Return confidence status based on thresholds."""
        if self.confidence_score >= 0.9:
            return "HIGH - Auto-approve"
        elif self.confidence_score >= 0.7:
            return "MEDIUM - Review recommended"
        else:
            return "LOW - Manual review required"


class RawExtraction(BaseModel):
    """
    Intermediate schema for raw extracted data before normalization.

    More lenient than StandardizedOrder to capture imperfect extractions.
    """

    order_id: Optional[str] = None
    client_name: Optional[str] = None
    order_date: Optional[str] = None
    delivery_date: Optional[str] = None
    items: list[dict] = Field(default_factory=list)
    order_total: Optional[float] = None
    currency: Optional[str] = None
    special_instructions: Optional[str] = None
    source_confidence: float = Field(
        default=1.0, description="Confidence from source parser"
    )
    extraction_method: str = Field(
        default="unknown", description="Method used for extraction"
    )
