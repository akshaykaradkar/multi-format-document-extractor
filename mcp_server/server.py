"""
FastMCP Server for Document Processing

Exposes document processing tools via the Model Context Protocol (MCP).
This allows AI agents and LLMs to interact with the document processing
pipeline through standardized tool calling.

Author: Akshay Karadkar
"""

import json
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    # Fallback for when fastmcp is not installed
    FastMCP = None

from .tools import DocumentTools


# Initialize MCP server
mcp = FastMCP("document-processor") if FastMCP else None

# Shared tools instance
_tools = DocumentTools(verbose=False)


if mcp:
    @mcp.tool()
    def process_document(file_path: str, mode: str = "hybrid") -> str:
        """
        Process a single document and extract order data.

        Args:
            file_path: Path to the document file (can be relative to sample_data/)
            mode: Extraction mode - 'rule_based', 'ai_only', or 'hybrid'

        Returns:
            JSON string with extraction results including order data, metrics,
            and confidence status.
        """
        result = _tools.process_document(file_path, mode)
        return json.dumps(result, indent=2, default=str)


    @mcp.tool()
    def list_sample_files() -> str:
        """
        List all available sample files for processing.

        Returns:
            JSON string with list of files including name, path, size, and extension.
        """
        result = _tools.list_sample_files()
        return json.dumps(result, indent=2)


    @mcp.tool()
    def get_confidence_report(file_path: str) -> str:
        """
        Get detailed confidence analysis for a document.

        Args:
            file_path: Path to the document file

        Returns:
            JSON string with confidence breakdown, recommendations, and order summary.
        """
        result = _tools.get_confidence_report(file_path)
        return json.dumps(result, indent=2, default=str)


    @mcp.tool()
    def compare_extraction_modes(file_path: str) -> str:
        """
        Compare all three extraction modes (rule_based, ai_only, hybrid) for a document.

        Args:
            file_path: Path to the document file

        Returns:
            JSON string comparing each mode's confidence, speed, cost, and a recommendation.
        """
        result = _tools.compare_extraction_modes(file_path)
        return json.dumps(result, indent=2, default=str)


    @mcp.tool()
    def get_extraction_stats(file_paths: Optional[str] = None) -> str:
        """
        Get batch extraction statistics for multiple files.

        Args:
            file_paths: Comma-separated list of file paths. If not provided,
                       processes all sample files.

        Returns:
            JSON string with batch statistics and individual results.
        """
        paths = None
        if file_paths:
            paths = [p.strip() for p in file_paths.split(",")]

        result = _tools.get_extraction_stats(paths)
        return json.dumps(result, indent=2, default=str)


def run_server():
    """Run the MCP server."""
    if not mcp:
        print("ERROR: MCP server not available. Install fastmcp:")
        print("  pip install fastmcp")
        sys.exit(1)

    mcp.run()


# For direct execution
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Test mode - run tools directly
        print("Testing Document Processing Tools\n")
        print("=" * 50)

        tools = DocumentTools(verbose=True)

        print("\n1. List Sample Files:")
        files = tools.list_sample_files()
        for f in files.get("files", []):
            print(f"   - {f['name']} ({f['size_kb']} KB)")

        print("\n2. Process First File (hybrid mode):")
        if files.get("files"):
            result = tools.process_document(
                files["files"][0]["path"],
                mode="hybrid"
            )
            if result.get("success"):
                print(f"   Order ID: {result['order']['order_id']}")
                print(f"   Client: {result['order']['client_name']}")
                print(f"   Confidence: {result['metrics']['confidence']:.2f}")
            else:
                print(f"   Error: {result.get('error')}")

        print("\n3. Get Extraction Stats:")
        stats = tools.get_extraction_stats()
        summary = stats.get("summary", {})
        print(f"   Total files: {summary.get('total_files')}")
        print(f"   Success rate: {summary.get('success_rate')}")
        print(f"   Avg time: {summary.get('avg_processing_time_ms'):.0f}ms")

    else:
        # Normal server mode
        run_server()
