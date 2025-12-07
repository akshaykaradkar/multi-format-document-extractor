#!/usr/bin/env python3
"""
Document Automation & Data Harmonization - Demo Script

This script demonstrates the end-to-end document processing pipeline.

Usage:
    python main.py                     # Process all sample files
    python main.py --file <path>       # Process single file
    python main.py --create-samples    # Create sample files first

Author: Akshay Karadkar
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import SAMPLE_DATA_DIR, OUTPUT_DIR, validate_config
from src.pipeline import DocumentPipeline, process_document


def print_banner():
    """Print application banner."""
    banner = """
========================================================================
|                                                                      |
|     DOCUMENT AUTOMATION & DATA HARMONIZATION SYSTEM                  |
|     ------------------------------------------------                 |
|     AI-Powered Purchase Order Processing PoC                         |
|                                                                      |
|     Author: Akshay Karadkar                                          |
|     Position: Senior AI Engineer                                     |
|                                                                      |
========================================================================
"""
    print(banner)


def print_section(title: str):
    """Print section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def create_samples():
    """Create sample files if they don't exist."""
    print_section("Creating Sample Files")

    try:
        from create_sample_files import main as create_main
        create_main()
        return True
    except Exception as e:
        print(f"Error creating samples: {e}")
        print("\nPlease run: pip install reportlab pandas openpyxl python-docx")
        return False


def process_all_samples(verbose: bool = True):
    """Process all sample files in the sample_data directory."""
    print_section("Processing All Sample Documents")

    # Check for sample files
    sample_files = list(SAMPLE_DATA_DIR.glob("*"))
    sample_files = [f for f in sample_files if f.suffix.lower() in
                    ['.pdf', '.xlsx', '.xls', '.docx', '.csv', '.jpg', '.jpeg', '.png']]

    if not sample_files:
        print("\nNo sample files found in sample_data/")
        print("Run with --create-samples first to generate test files.")
        return []

    print(f"\nFound {len(sample_files)} sample file(s):")
    for f in sample_files:
        print(f"  - {f.name}")

    # Process each file
    pipeline = DocumentPipeline()
    results = []

    for file_path in sample_files:
        print(f"\n{'-'*70}")
        result = pipeline.process(file_path, save_output=True, verbose=verbose)
        results.append(result)

    return results


def print_results_summary(results: list):
    """Print summary of all processed documents."""
    print_section("Processing Summary")

    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    print(f"\n  Total Processed:  {len(results)}")
    print(f"  Successful:       {len(successful)}")
    print(f"  Failed:           {len(failed)}")

    if successful:
        print("\n  [OK] Successfully Processed Orders:")
        print("  " + "-"*66)
        print(f"  {'Order ID':<20} {'Client':<25} {'Total':>10} {'Confidence':>10}")
        print("  " + "-"*66)

        for r in successful:
            order = r.get("order", {})
            conf = r.get("confidence", 0)
            status = r.get("confidence_status", {}).get("status", "")
            print(f"  {order.get('order_id', 'N/A'):<20} "
                  f"{order.get('client_name', 'N/A')[:25]:<25} "
                  f"${order.get('order_total', 0):>9,.2f} "
                  f"{conf:.2f} ({status})")

    if failed:
        print("\n  [FAIL] Failed:")
        for r in failed:
            print(f"    - {r.get('file', 'Unknown')}: {r.get('error', 'Unknown error')}")

    # Output location
    print(f"\n  Output files saved to: {OUTPUT_DIR}/")


def process_single_file(file_path: str, verbose: bool = True):
    """Process a single file."""
    print_section(f"Processing: {Path(file_path).name}")

    result = process_document(file_path, save_output=True, verbose=verbose)

    if result.get("success"):
        print("\n[OK] Processing successful!")
        print(f"\nOutput saved to: {result.get('output_file')}")

        # Print full JSON output
        print("\n" + "-"*70)
        print("JSON Output:")
        print("-"*70)
        print(json.dumps(result.get("order"), indent=2))
    else:
        print(f"\n[FAIL] Processing failed: {result.get('error')}")

    return result


def run_demo():
    """Run the full demonstration."""
    print_banner()

    # Validate configuration
    print_section("Configuration Check")
    config_valid = validate_config()
    if config_valid:
        print("  [OK] OpenAI API key configured")
    else:
        print("  [WARN] OpenAI API key not set - OCR features disabled")
        print("    Set OPENAI_API_KEY in .env file for full functionality")

    # Check for sample files
    print_section("Sample Files Check")
    sample_files = list(SAMPLE_DATA_DIR.glob("*"))
    valid_extensions = ['.pdf', '.xlsx', '.xls', '.docx', '.csv', '.jpg', '.jpeg', '.png']
    sample_files = [f for f in sample_files if f.suffix.lower() in valid_extensions]

    if len(sample_files) < 4:
        print(f"  Found {len(sample_files)} sample file(s)")
        print("  Creating additional sample files...")
        create_samples()

    # Process all samples
    results = process_all_samples(verbose=True)

    # Print summary
    if results:
        print_results_summary(results)

    # Final message
    print_section("Demo Complete")
    print("""
  This PoC demonstrates:

  1. Multi-format Document Parsing
     - PDF (pdfplumber)
     - Excel multi-sheet (pandas + openpyxl)
     - Word documents (python-docx)
     - CSV with varying structures (pandas)
     - Scanned/handwritten forms (GPT-4o Vision OCR)

  2. Intelligent Data Extraction
     - Hybrid approach: local parsing + LLM
     - Automatic field mapping
     - Date normalization

  3. Quality Assurance
     - Confidence scoring (0.0-1.0)
     - Schema validation (Pydantic)
     - Human-in-the-loop flagging

  4. Standardized Output
     - JSON schema as specified
     - Validation reports
     - Audit trail

  For more details, see the technical documentation in docs/
""")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Document Automation & Data Harmonization PoC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Process a single file"
    )
    parser.add_argument(
        "--create-samples",
        action="store_true",
        help="Create sample test files"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    if args.create_samples:
        print_banner()
        create_samples()
    elif args.file:
        print_banner()
        process_single_file(args.file, verbose=not args.quiet)
    else:
        run_demo()


if __name__ == "__main__":
    main()
