"""
MCP Tools for Document Processing

This module wraps the existing HybridPipeline with MCP-compatible tools.
No modifications to the original pipeline - purely additive.

Author: Akshay Karadkar
"""

import json
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SAMPLE_DATA_DIR, OUTPUT_DIR
from src.hybrid_pipeline import HybridPipeline, ExtractionMode


class DocumentTools:
    """
    Document processing tools for MCP server.

    Wraps HybridPipeline functionality as discrete tools that can be
    called by AI agents through the MCP protocol.
    """

    def __init__(self, verbose: bool = False):
        """Initialize tools with optional verbose mode."""
        self.verbose = verbose
        self._pipeline_cache = {}

    def _get_pipeline(self, mode: str = "hybrid") -> HybridPipeline:
        """Get or create pipeline for specified mode."""
        if mode not in self._pipeline_cache:
            mode_enum = ExtractionMode(mode)
            self._pipeline_cache[mode] = HybridPipeline(
                mode=mode_enum,
                verbose=self.verbose
            )
        return self._pipeline_cache[mode]

    def process_document(
        self,
        file_path: str,
        mode: str = "hybrid"
    ) -> dict:
        """
        Process a single document and extract order data.

        Args:
            file_path: Path to the document file
            mode: Extraction mode - 'rule_based', 'ai_only', or 'hybrid'

        Returns:
            Dictionary with extraction results including:
            - success: bool
            - order: Standardized order data (if successful)
            - metrics: Processing metrics
            - confidence_status: Confidence level interpretation
        """
        path = Path(file_path)

        # Handle relative paths - check sample_data directory
        if not path.is_absolute():
            sample_path = SAMPLE_DATA_DIR / file_path
            if sample_path.exists():
                path = sample_path

        if not path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "suggestion": "Use list_sample_files() to see available files"
            }

        # Validate mode
        valid_modes = ["rule_based", "ai_only", "hybrid"]
        if mode not in valid_modes:
            return {
                "success": False,
                "error": f"Invalid mode: {mode}",
                "valid_modes": valid_modes
            }

        try:
            pipeline = self._get_pipeline(mode)
            result = pipeline.process(path, save_output=True)
            return result
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file": str(path)
            }

    def list_sample_files(self) -> dict:
        """
        List all available sample files for processing.

        Returns:
            Dictionary with:
            - files: List of file information
            - directory: Sample data directory path
        """
        files = []

        if SAMPLE_DATA_DIR.exists():
            for f in sorted(SAMPLE_DATA_DIR.iterdir()):
                if f.is_file() and not f.name.startswith('.'):
                    files.append({
                        "name": f.name,
                        "path": str(f),
                        "size_kb": round(f.stat().st_size / 1024, 2),
                        "extension": f.suffix.lower()
                    })

        return {
            "directory": str(SAMPLE_DATA_DIR),
            "file_count": len(files),
            "files": files
        }

    def get_confidence_report(self, file_path: str) -> dict:
        """
        Get detailed confidence analysis for a document.

        Args:
            file_path: Path to the document file

        Returns:
            Dictionary with confidence breakdown and recommendations
        """
        # First process the document
        result = self.process_document(file_path, mode="hybrid")

        if not result.get("success"):
            return result

        metrics = result.get("metrics", {})
        confidence = metrics.get("confidence", 0)

        # Build confidence report
        report = {
            "file": file_path,
            "overall_confidence": confidence,
            "confidence_status": result.get("confidence_status", "Unknown"),
            "recommendation": self._get_recommendation(confidence),
            "breakdown": {
                "extraction_method": metrics.get("method_used", "unknown"),
                "fields_extracted": metrics.get("fields_extracted", 0),
                "total_fields": metrics.get("total_fields", 6),
                "processing_time_ms": metrics.get("processing_time_ms", 0),
                "estimated_cost_usd": metrics.get("estimated_cost", 0)
            }
        }

        # Add order summary if available
        if result.get("order"):
            order = result["order"]
            report["order_summary"] = {
                "order_id": order.get("order_id"),
                "client_name": order.get("client_name"),
                "item_count": len(order.get("items", [])),
                "order_total": order.get("order_total")
            }

        return report

    def _get_recommendation(self, confidence: float) -> str:
        """Get recommendation based on confidence score."""
        if confidence >= 0.9:
            return "AUTO_APPROVE - High confidence extraction"
        elif confidence >= 0.7:
            return "REVIEW_RECOMMENDED - Some fields may need verification"
        else:
            return "MANUAL_REVIEW_REQUIRED - Low confidence, human review needed"

    def compare_extraction_modes(self, file_path: str) -> dict:
        """
        Compare all three extraction modes for a document.

        Args:
            file_path: Path to the document file

        Returns:
            Dictionary comparing rule_based, ai_only, and hybrid modes
        """
        results = {}

        for mode in ["rule_based", "ai_only", "hybrid"]:
            result = self.process_document(file_path, mode=mode)

            if result.get("success"):
                metrics = result.get("metrics", {})
                results[mode] = {
                    "success": True,
                    "confidence": metrics.get("confidence", 0),
                    "processing_time_ms": metrics.get("processing_time_ms", 0),
                    "method_used": metrics.get("method_used", "unknown"),
                    "estimated_cost_usd": metrics.get("estimated_cost", 0),
                    "fields_extracted": metrics.get("fields_extracted", 0)
                }
            else:
                results[mode] = {
                    "success": False,
                    "error": result.get("error", "Unknown error")
                }

        # Add comparison summary
        successful_modes = [m for m, r in results.items() if r.get("success")]

        if successful_modes:
            best_confidence = max(
                successful_modes,
                key=lambda m: results[m].get("confidence", 0)
            )
            fastest = min(
                successful_modes,
                key=lambda m: results[m].get("processing_time_ms", float('inf'))
            )
            cheapest = min(
                successful_modes,
                key=lambda m: results[m].get("estimated_cost_usd", float('inf'))
            )

            results["comparison"] = {
                "best_confidence": best_confidence,
                "fastest": fastest,
                "cheapest": cheapest,
                "recommendation": self._recommend_mode(results)
            }

        return results

    def _recommend_mode(self, results: dict) -> str:
        """Recommend best mode based on comparison results."""
        hybrid = results.get("hybrid", {})
        rule_based = results.get("rule_based", {})
        ai_only = results.get("ai_only", {})

        # If rule_based has high confidence, use it (free and fast)
        if rule_based.get("success") and rule_based.get("confidence", 0) >= 0.9:
            return "rule_based - Free, fast, and high confidence"

        # If hybrid succeeded with good confidence, recommend it
        if hybrid.get("success") and hybrid.get("confidence", 0) >= 0.7:
            return "hybrid - Good balance of cost and accuracy"

        # Fall back to ai_only if needed
        if ai_only.get("success"):
            return "ai_only - Best for complex/scanned documents"

        return "hybrid - Default recommendation"

    def get_extraction_stats(self, file_paths: Optional[list] = None) -> dict:
        """
        Get batch extraction statistics for multiple files.

        Args:
            file_paths: List of file paths. If None, processes all sample files.

        Returns:
            Dictionary with batch statistics and individual results
        """
        # Default to sample files if none provided
        if file_paths is None:
            sample_files = self.list_sample_files()
            file_paths = [f["path"] for f in sample_files.get("files", [])]

        if not file_paths:
            return {
                "success": False,
                "error": "No files to process"
            }

        results = []
        total_time = 0
        total_cost = 0
        success_count = 0

        for file_path in file_paths:
            result = self.process_document(file_path, mode="hybrid")

            if result.get("success"):
                metrics = result.get("metrics", {})
                success_count += 1
                total_time += metrics.get("processing_time_ms", 0)
                total_cost += metrics.get("estimated_cost", 0)

                results.append({
                    "file": Path(file_path).name,
                    "success": True,
                    "confidence": metrics.get("confidence", 0),
                    "processing_time_ms": metrics.get("processing_time_ms", 0),
                    "method": metrics.get("method_used", "unknown")
                })
            else:
                results.append({
                    "file": Path(file_path).name,
                    "success": False,
                    "error": result.get("error", "Unknown error")
                })

        return {
            "summary": {
                "total_files": len(file_paths),
                "successful": success_count,
                "failed": len(file_paths) - success_count,
                "success_rate": f"{(success_count / len(file_paths) * 100):.1f}%",
                "total_processing_time_ms": round(total_time, 2),
                "avg_processing_time_ms": round(total_time / len(file_paths), 2) if file_paths else 0,
                "total_estimated_cost_usd": round(total_cost, 4)
            },
            "results": results
        }


# For direct testing
if __name__ == "__main__":
    tools = DocumentTools(verbose=True)

    print("\n=== List Sample Files ===")
    files = tools.list_sample_files()
    print(json.dumps(files, indent=2))

    print("\n=== Process Single Document ===")
    if files.get("files"):
        first_file = files["files"][0]["path"]
        result = tools.process_document(first_file, mode="hybrid")
        print(json.dumps(result, indent=2, default=str))
