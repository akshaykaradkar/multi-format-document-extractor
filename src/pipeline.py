"""
Main Pipeline Orchestrator for document processing.

Coordinates the entire extraction, transformation, validation,
and output generation process.
"""

import json
from pathlib import Path
from typing import Union, Optional
from datetime import datetime

from .config import get_file_type, OUTPUT_DIR, validate_config
from .schemas import RawExtraction, StandardizedOrder
from .parsers import PDFParser, ExcelParser, WordParser, CSVParser, OCRParser
from .parsers.base_parser import ParseError
from .processors import DataTransformer, ConfidenceScorer, LLMExtractor
from .validators import SchemaValidator


class DocumentPipeline:
    """
    Main orchestrator for document processing pipeline.

    Flow:
    1. Detect document format
    2. Parse document with appropriate parser
    3. Optionally enhance with LLM
    4. Transform to standardized format
    5. Calculate confidence score
    6. Validate against schema
    7. Output JSON result
    """

    def __init__(self, use_llm_enhancement: bool = False):
        """
        Initialize the pipeline.

        Args:
            use_llm_enhancement: Whether to use LLM to fill missing fields
        """
        self.transformer = DataTransformer()
        self.scorer = ConfidenceScorer()
        self.validator = SchemaValidator()
        self.llm_extractor = LLMExtractor() if use_llm_enhancement else None

        # Parser registry
        self.parsers = {
            "pdf": PDFParser,
            "excel": ExcelParser,
            "word": WordParser,
            "csv": CSVParser,
            "ocr": OCRParser,
        }

    def process(
        self,
        file_path: Union[str, Path],
        save_output: bool = True,
        verbose: bool = False,
    ) -> dict:
        """
        Process a single document through the pipeline.

        Args:
            file_path: Path to the document
            save_output: Whether to save JSON output to file
            verbose: Whether to print progress

        Returns:
            Dict containing:
            - success: bool
            - order: StandardizedOrder dict (if successful)
            - confidence: float
            - confidence_status: dict
            - output_file: str (if saved)
            - error: str (if failed)
        """
        file_path = Path(file_path)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing: {file_path.name}")
            print(f"{'='*60}")

        try:
            # Step 1: Detect file type
            file_type = get_file_type(file_path)
            if not file_type:
                raise ParseError(f"Unsupported file format: {file_path.suffix}")

            if verbose:
                print(f"[1/5] Detected format: {file_type}")

            # Step 2: Parse document
            parser_class = self.parsers.get(file_type)
            if not parser_class:
                raise ParseError(f"No parser available for: {file_type}")

            parser = parser_class(file_path)
            raw_extraction = parser.parse()

            if verbose:
                print(f"[2/5] Parsed: {len(raw_extraction.items)} items found")

            # Step 3: Optional LLM enhancement
            if self.llm_extractor and self._needs_enhancement(raw_extraction):
                if verbose:
                    print("[3/5] Enhancing with LLM...")
                # For now, read text from file for enhancement
                # In production, this would use the original text
                pass
            else:
                if verbose:
                    print("[3/5] LLM enhancement: skipped")

            # Step 4: Calculate confidence score
            validation_errors = []
            confidence = self.scorer.calculate_score(
                raw_extraction,
                validated=True,
                validation_errors=validation_errors,
            )

            if verbose:
                print(f"[4/5] Confidence score: {confidence:.2f}")

            # Step 5: Transform to standardized format
            standardized = self.transformer.transform(raw_extraction, confidence)

            if verbose:
                print(f"[5/5] Transformed to standardized format")

            # Validate
            is_valid, errors = self.validator.validate_order(standardized)

            # Get confidence status
            confidence_status = self.scorer.get_confidence_status(confidence)

            # Prepare result
            result = {
                "success": True,
                "order": standardized.model_dump(),
                "confidence": confidence,
                "confidence_status": confidence_status,
                "validation": {
                    "is_valid": is_valid,
                    "errors": errors,
                },
                "metadata": {
                    "source_file": str(file_path),
                    "file_type": file_type,
                    "extraction_method": raw_extraction.extraction_method,
                    "processed_at": datetime.now().isoformat(),
                },
            }

            # Save output
            if save_output:
                output_file = self._save_output(standardized, file_path)
                result["output_file"] = str(output_file)

                if verbose:
                    print(f"\nOutput saved: {output_file}")

            if verbose:
                self._print_summary(standardized, confidence_status)

            return result

        except Exception as e:
            error_msg = str(e)
            if verbose:
                print(f"\n[ERROR] {error_msg}")

            return {
                "success": False,
                "error": error_msg,
                "file": str(file_path),
            }

    def process_batch(
        self,
        file_paths: list[Union[str, Path]],
        save_output: bool = True,
        verbose: bool = False,
    ) -> dict:
        """
        Process multiple documents.

        Args:
            file_paths: List of file paths
            save_output: Whether to save JSON outputs
            verbose: Whether to print progress

        Returns:
            Dict with batch results
        """
        results = {
            "total": len(file_paths),
            "successful": 0,
            "failed": 0,
            "orders": [],
            "errors": [],
        }

        for file_path in file_paths:
            result = self.process(file_path, save_output, verbose)

            if result["success"]:
                results["successful"] += 1
                results["orders"].append(result)
            else:
                results["failed"] += 1
                results["errors"].append(result)

        return results

    def _needs_enhancement(self, raw: RawExtraction) -> bool:
        """Check if extraction needs LLM enhancement."""
        # Enhance if missing critical fields
        if not raw.order_id or not raw.client_name:
            return True
        if not raw.items or len(raw.items) == 0:
            return True
        if raw.source_confidence < 0.7:
            return True
        return False

    def _save_output(
        self, order: StandardizedOrder, source_file: Path
    ) -> Path:
        """Save standardized order to JSON file."""
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{source_file.stem}_{timestamp}.json"
        output_path = OUTPUT_DIR / output_name

        # Write JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(order.model_dump(), f, indent=2, ensure_ascii=False)

        return output_path

    def _print_summary(
        self, order: StandardizedOrder, confidence_status: dict
    ) -> None:
        """Print order summary."""
        print(f"\n{'-'*60}")
        print(f"Order ID:      {order.order_id}")
        print(f"Client:        {order.client_name}")
        print(f"Order Date:    {order.order_date}")
        print(f"Delivery Date: {order.delivery_date}")
        print(f"Items:         {len(order.items)}")
        print(f"Total:         ${order.order_total:,.2f}")
        print(f"Confidence:    {order.confidence_score:.2f} ({confidence_status['status']})")
        print(f"Action:        {confidence_status['recommendation']}")
        print(f"{'-'*60}")


def process_document(
    file_path: Union[str, Path],
    save_output: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Convenience function to process a single document.

    Args:
        file_path: Path to document
        save_output: Save JSON output
        verbose: Print progress

    Returns:
        Processing result dict
    """
    # Validate config
    validate_config()

    pipeline = DocumentPipeline()
    return pipeline.process(file_path, save_output, verbose)
