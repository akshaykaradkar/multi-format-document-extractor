"""
LLM Extractor for intelligent field extraction.

Uses OpenAI's structured outputs to extract and map fields
from unstructured or semi-structured text content.
"""

import json
from typing import Optional

from ..schemas import RawExtraction
from ..config import OPENAI_API_KEY, OPENAI_MODEL


class LLMExtractor:
    """
    Uses LLM for intelligent field extraction when local parsing
    produces incomplete or uncertain results.

    Features:
    - Structured output extraction with Pydantic schemas
    - Field mapping from various naming conventions
    - Context-aware extraction
    """

    def __init__(self):
        """Initialize the LLM extractor."""
        self.client = None
        if OPENAI_API_KEY:
            from openai import OpenAI

            self.client = OpenAI(api_key=OPENAI_API_KEY)

    def extract_from_text(self, text: str) -> RawExtraction:
        """
        Extract order data from unstructured text using LLM.

        Args:
            text: Raw text content from document

        Returns:
            RawExtraction with extracted data
        """
        if not self.client:
            raise RuntimeError("OpenAI API key not configured")

        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """You are a document data extraction assistant.
Extract order information from the provided text and return it as JSON.
Be precise and extract only what is explicitly stated in the document.
If a field is not found, use null.""",
                },
                {
                    "role": "user",
                    "content": f"""Extract order information from this text:

{text}

Return JSON with this structure:
{{
    "order_id": "string or null",
    "client_name": "string or null",
    "order_date": "YYYY-MM-DD or null",
    "delivery_date": "YYYY-MM-DD or null",
    "items": [
        {{
            "product_code": "string",
            "description": "string",
            "quantity": number,
            "unit_price": number,
            "total_price": number
        }}
    ],
    "order_total": number or null,
    "currency": "USD/EUR/etc or null",
    "special_instructions": "string or null"
}}

Return ONLY valid JSON, no markdown or explanation.""",
                },
            ],
            max_tokens=2000,
            temperature=0.1,
        )

        content = response.choices[0].message.content
        return self._parse_response(content)

    def enhance_extraction(
        self, raw: RawExtraction, original_text: str
    ) -> RawExtraction:
        """
        Enhance a partial extraction by filling in missing fields.

        Args:
            raw: Partial RawExtraction from parser
            original_text: Original document text

        Returns:
            Enhanced RawExtraction
        """
        if not self.client:
            return raw

        # Identify missing fields
        missing = []
        if not raw.order_id:
            missing.append("order_id")
        if not raw.client_name:
            missing.append("client_name")
        if not raw.order_date:
            missing.append("order_date")
        if not raw.delivery_date:
            missing.append("delivery_date")
        if not raw.items:
            missing.append("items")

        if not missing:
            return raw

        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """You are a document data extraction assistant.
Help fill in missing order fields from the document text.""",
                },
                {
                    "role": "user",
                    "content": f"""I have partially extracted order data but need help with these fields: {', '.join(missing)}

Already extracted:
- Order ID: {raw.order_id}
- Client: {raw.client_name}
- Order Date: {raw.order_date}
- Delivery Date: {raw.delivery_date}
- Items: {len(raw.items)} found
- Total: {raw.order_total}

Original document text:
{original_text[:3000]}

Please extract ONLY the missing fields and return as JSON:
{{
    "order_id": "string or null if not in missing list",
    "client_name": "string or null if not in missing list",
    "order_date": "YYYY-MM-DD or null",
    "delivery_date": "YYYY-MM-DD or null",
    "items": [...] or null if not in missing list
}}

Return ONLY the fields that need to be filled in.""",
                },
            ],
            max_tokens=1500,
            temperature=0.1,
        )

        content = response.choices[0].message.content

        try:
            # Clean and parse response
            content = content.strip()
            if content.startswith("```"):
                import re

                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)

            enhanced_data = json.loads(content)

            # Merge with existing data
            if not raw.order_id and enhanced_data.get("order_id"):
                raw.order_id = enhanced_data["order_id"]
            if not raw.client_name and enhanced_data.get("client_name"):
                raw.client_name = enhanced_data["client_name"]
            if not raw.order_date and enhanced_data.get("order_date"):
                raw.order_date = enhanced_data["order_date"]
            if not raw.delivery_date and enhanced_data.get("delivery_date"):
                raw.delivery_date = enhanced_data["delivery_date"]
            if not raw.items and enhanced_data.get("items"):
                raw.items = enhanced_data["items"]

        except (json.JSONDecodeError, KeyError):
            # If enhancement fails, return original
            pass

        return raw

    def _parse_response(self, content: str) -> RawExtraction:
        """Parse LLM response into RawExtraction."""
        import re

        # Clean up response
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try to find JSON in response
            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                data = json.loads(json_match.group(0))
            else:
                raise ValueError("Could not parse LLM response as JSON")

        return RawExtraction(
            order_id=data.get("order_id"),
            client_name=data.get("client_name"),
            order_date=data.get("order_date"),
            delivery_date=data.get("delivery_date"),
            items=data.get("items", []),
            order_total=data.get("order_total"),
            currency=data.get("currency"),
            special_instructions=data.get("special_instructions"),
            source_confidence=0.85,
            extraction_method="llm_openai",
        )
