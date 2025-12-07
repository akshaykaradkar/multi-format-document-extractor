"""
OCR Parser for scanned/handwritten documents.

Handles Client E (Local Hardware Co) - Scanned PDF or image
of handwritten/typed order forms using GPT-4o Vision.
"""

import base64
import re
from pathlib import Path
from typing import Union

from .base_parser import BaseParser, ParseError
from ..schemas import RawExtraction
from ..config import OPENAI_API_KEY, OPENAI_VISION_MODEL


class OCRParser(BaseParser):
    """
    Parser for scanned documents and images using GPT-4o Vision.

    Uses OpenAI's GPT-4o Vision API for OCR and intelligent
    field extraction from handwritten or typed forms.
    """

    @property
    def supported_extensions(self) -> list[str]:
        return [".jpg", ".jpeg", ".png", ".gif", ".webp"]

    def parse(self) -> RawExtraction:
        """
        Parse image using GPT-4o Vision for OCR.

        Returns:
            RawExtraction with parsed order data
        """
        if not OPENAI_API_KEY:
            raise ParseError(
                "OpenAI API key not configured. Set OPENAI_API_KEY environment variable.",
                file_path=self.file_path,
            )

        try:
            # Read and encode image
            image_data = self._encode_image()

            # Call GPT-4o Vision
            extraction = self._extract_with_vision(image_data)

            return extraction

        except Exception as e:
            raise ParseError(
                f"Failed to parse image with OCR: {str(e)}",
                file_path=self.file_path,
                original_error=e,
            )

    def _encode_image(self) -> str:
        """Encode image to base64."""
        with open(self.file_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    def _get_media_type(self) -> str:
        """Get MIME type for image."""
        ext = self.file_path.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return mime_types.get(ext, "image/jpeg")

    def _extract_with_vision(self, image_base64: str) -> RawExtraction:
        """
        Use GPT-4o Vision to extract order data from image.

        Args:
            image_base64: Base64 encoded image

        Returns:
            RawExtraction object
        """
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)

        # Construct the prompt for structured extraction
        extraction_prompt = """Analyze this order form image and extract the following information in JSON format:

{
    "order_id": "the order number/ID",
    "client_name": "company name from the form header",
    "order_date": "date in YYYY-MM-DD format",
    "delivery_date": "need by/delivery date in YYYY-MM-DD format",
    "delivery_type": "standard or rush",
    "items": [
        {
            "product_code": "generate a code if not visible",
            "description": "item description",
            "quantity": numeric quantity,
            "unit_price": numeric price per unit,
            "total_price": quantity * unit_price
        }
    ],
    "special_instructions": "any special notes or instructions",
    "contact_info": "contact name and phone if visible"
}

Important:
- Convert all dates to YYYY-MM-DD format
- Extract ALL items visible on the form
- Parse handwritten text carefully
- If a checkbox is marked (X or checked), note which option is selected
- Include any special delivery instructions
- Return ONLY valid JSON, no markdown or explanation"""

        response = client.chat.completions.create(
            model=OPENAI_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": extraction_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{self._get_media_type()};base64,{image_base64}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=2000,
            temperature=0.1,  # Low temperature for accuracy
        )

        # Parse the response
        content = response.choices[0].message.content
        return self._parse_vision_response(content)

    def _parse_vision_response(self, content: str) -> RawExtraction:
        """
        Parse GPT-4o Vision response into RawExtraction.

        Args:
            content: JSON string from Vision API

        Returns:
            RawExtraction object
        """
        import json

        # Clean up response - remove markdown code blocks if present
        content = content.strip()
        if content.startswith("```"):
            # Remove ```json and ``` markers
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            # Try to extract JSON from response
            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                data = json.loads(json_match.group(0))
            else:
                raise ParseError(f"Failed to parse Vision API response as JSON: {e}")

        # Extract items
        items = []
        for item_data in data.get("items", []):
            item = {
                "product_code": item_data.get("product_code", "ITEM"),
                "description": item_data.get("description", ""),
                "quantity": self._safe_int(item_data.get("quantity", 1)),
                "unit_price": self._safe_float(item_data.get("unit_price", 0)),
                "total_price": self._safe_float(item_data.get("total_price", 0)),
            }
            # Calculate total if not provided
            if item["total_price"] == 0:
                item["total_price"] = item["quantity"] * item["unit_price"]
            if item["description"]:
                items.append(item)

        # Build special instructions
        instructions = []
        if data.get("special_instructions"):
            instructions.append(data["special_instructions"])
        if data.get("contact_info"):
            instructions.append(f"Contact: {data['contact_info']}")
        if data.get("delivery_type", "").lower() == "rush":
            instructions.append("Rush delivery requested")

        # Normalize dates
        order_date = self._normalize_date(data.get("order_date"))
        delivery_date = self._normalize_date(data.get("delivery_date"))

        # Calculate total
        order_total = sum(item["total_price"] for item in items)

        return RawExtraction(
            order_id=data.get("order_id"),
            client_name=data.get("client_name"),
            order_date=order_date,
            delivery_date=delivery_date,
            items=items,
            order_total=order_total,
            currency="USD",
            special_instructions="; ".join(instructions) if instructions else None,
            source_confidence=0.80,  # Medium confidence for OCR
            extraction_method="ocr_gpt4o_vision",
        )

    def _normalize_date(self, date_str: str | None) -> str | None:
        """Normalize date to YYYY-MM-DD format."""
        if not date_str:
            return None

        from dateutil import parser

        try:
            parsed = parser.parse(str(date_str))
            return parsed.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            # Check if already in correct format
            if re.match(r"\d{4}-\d{2}-\d{2}", str(date_str)):
                return str(date_str)
            return None

    def _safe_int(self, value) -> int:
        """Safely convert to integer."""
        try:
            if isinstance(value, str):
                # Remove non-numeric characters except digits
                cleaned = re.sub(r"[^\d]", "", value)
                return int(cleaned) if cleaned else 1
            return int(value)
        except (ValueError, TypeError):
            return 1

    def _safe_float(self, value) -> float:
        """Safely convert to float."""
        try:
            if isinstance(value, str):
                # Remove currency symbols and commas
                cleaned = re.sub(r"[^\d.]", "", value)
                return float(cleaned) if cleaned else 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0
