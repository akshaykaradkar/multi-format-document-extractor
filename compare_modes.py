#!/usr/bin/env python3
"""
Extraction Mode Comparison Demo

Compares 3 extraction approaches:
1. rule_based - Fast, free, works for structured documents
2. ai_only    - Uses AI models for all documents
3. hybrid     - Intelligent routing: rule-based first, AI fallback

This demonstrates Senior AI Engineer judgment:
- Knowing WHEN to use AI vs rule-based
- Cost-benefit analysis
- Intelligent resource allocation

Usage:
    python compare_modes.py                    # Compare all modes on all files
    python compare_modes.py --file <path>      # Compare on single file
    python compare_modes.py --mode rule_based  # Run specific mode only

Author: Akshay Karadkar
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import SAMPLE_DATA_DIR, validate_config
from src.hybrid_pipeline import HybridPipeline, ExtractionMode


def print_banner():
    """Print application banner."""
    print("""
========================================================================
|                                                                      |
|     EXTRACTION MODE COMPARISON DEMO                                  |
|     --------------------------------                                 |
|                                                                      |
|     Comparing: rule_based vs ai_only vs hybrid                       |
|                                                                      |
|     This demonstrates WHY a Senior AI Engineer doesn't just          |
|     "use AI everywhere" - it's about intelligent resource use.       |
|                                                                      |
========================================================================
""")


def get_sample_files() -> List[Path]:
    """Get sample files for comparison."""
    valid_extensions = ['.pdf', '.xlsx', '.xls', '.docx', '.csv', '.jpg', '.jpeg', '.png']
    files = list(SAMPLE_DATA_DIR.glob("*"))
    return [f for f in files if f.suffix.lower() in valid_extensions]


def run_comparison(
    file_path: Path,
    modes: List[str],
    verbose: bool = False,
) -> Dict:
    """
    Run extraction with multiple modes on same file.

    Returns comparison results.
    """
    results = {}

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Testing: {mode.upper()}")
        print(f"{'='*60}")

        try:
            pipeline = HybridPipeline(
                mode=ExtractionMode(mode),
                verbose=verbose,
            )

            start = time.time()
            result = pipeline.process(file_path, save_output=False)
            elapsed = (time.time() - start) * 1000

            results[mode] = {
                "success": result.get("success", False),
                "time_ms": elapsed,
                "confidence": result.get("metrics", {}).get("confidence", 0),
                "method": result.get("metrics", {}).get("method_used", "unknown"),
                "cost": result.get("metrics", {}).get("estimated_cost", 0),
                "error": result.get("error"),
            }

            if result.get("success"):
                order = result.get("order", {})
                results[mode]["order_id"] = order.get("order_id", "N/A")
                results[mode]["client"] = order.get("client_name", "N/A")
                results[mode]["total"] = order.get("order_total", 0)

        except Exception as e:
            results[mode] = {
                "success": False,
                "error": str(e),
                "time_ms": 0,
                "confidence": 0,
            }

    return results


def print_comparison_table(file_name: str, results: Dict):
    """Print comparison table for a single file."""
    print(f"\n{'='*80}")
    print(f"COMPARISON: {file_name}")
    print(f"{'='*80}")

    # Header
    print(f"\n{'Mode':<15} {'Status':<10} {'Time':<12} {'Confidence':<12} {'Cost':<10} {'Method':<20}")
    print("-" * 80)

    for mode, data in results.items():
        status = "OK" if data.get("success") else "FAIL"
        time_str = f"{data.get('time_ms', 0):.0f}ms"
        conf_str = f"{data.get('confidence', 0):.2f}"
        cost_str = f"${data.get('cost', 0):.4f}"
        method = data.get("method", "N/A")[:20]

        print(f"{mode:<15} {status:<10} {time_str:<12} {conf_str:<12} {cost_str:<10} {method:<20}")

    # Winner analysis
    print("\n" + "-" * 80)
    print("ANALYSIS:")

    successful = {k: v for k, v in results.items() if v.get("success")}

    if successful:
        # Fastest
        fastest = min(successful, key=lambda x: successful[x].get("time_ms", float("inf")))
        print(f"  Fastest:     {fastest} ({successful[fastest]['time_ms']:.0f}ms)")

        # Most confident
        most_confident = max(successful, key=lambda x: successful[x].get("confidence", 0))
        print(f"  Best conf:   {most_confident} ({successful[most_confident]['confidence']:.2f})")

        # Cheapest
        cheapest = min(successful, key=lambda x: successful[x].get("cost", float("inf")))
        print(f"  Cheapest:    {cheapest} (${successful[cheapest]['cost']:.4f})")

        # Recommendation
        print("\n  RECOMMENDATION:")
        if "rule_based" in successful and successful["rule_based"]["confidence"] >= 0.9:
            print("  -> Use RULE_BASED: High confidence, zero cost, fastest")
        elif "hybrid" in successful:
            print("  -> Use HYBRID: Best of both worlds - fast for structured, AI for complex")
        elif "ai_only" in successful:
            print("  -> Use AI_ONLY: Only option that works for this document")
    else:
        print("  No successful extractions!")


def print_summary(all_results: Dict):
    """Print overall summary."""
    print("\n")
    print("=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    # Aggregate stats per mode
    modes = ["rule_based", "ai_only", "hybrid"]
    summary = {mode: {"success": 0, "fail": 0, "total_time": 0, "total_cost": 0} for mode in modes}

    for file_name, results in all_results.items():
        for mode, data in results.items():
            if mode in summary:
                if data.get("success"):
                    summary[mode]["success"] += 1
                else:
                    summary[mode]["fail"] += 1
                summary[mode]["total_time"] += data.get("time_ms", 0)
                summary[mode]["total_cost"] += data.get("cost", 0)

    print(f"\n{'Mode':<15} {'Success':<10} {'Failed':<10} {'Avg Time':<15} {'Total Cost':<12}")
    print("-" * 65)

    for mode in modes:
        s = summary[mode]
        total = s["success"] + s["fail"]
        avg_time = s["total_time"] / total if total > 0 else 0
        print(f"{mode:<15} {s['success']:<10} {s['fail']:<10} {avg_time:.0f}ms{'':<8} ${s['total_cost']:.4f}")

    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS FOR INTERVIEW:")
    print("=" * 80)
    print("""
1. RULE_BASED is fastest and free - use for structured documents
   - PDFs, Excel, Word, CSV with known formats
   - ~40-100ms per document, $0 cost

2. AI_ONLY provides best accuracy for complex documents
   - Scanned images, handwritten forms, unknown layouts
   - ~5-50 seconds per document, ~$0.005-0.02 per doc

3. HYBRID is the intelligent approach
   - Uses rule-based first (fast, free)
   - Falls back to AI only when needed
   - Best cost/accuracy tradeoff

4. Senior AI Engineer judgment:
   - "Use AI everywhere" = wasteful
   - "Use rule-based only" = brittle
   - "Route intelligently" = production-ready
""")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare extraction modes: rule_based vs ai_only vs hybrid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Process single file only"
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["rule_based", "ai_only", "hybrid", "all"],
        default="all",
        help="Run specific mode only (default: all)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    print_banner()

    # Validate config
    validate_config()

    # Determine modes to test
    if args.mode == "all":
        modes = ["rule_based", "ai_only", "hybrid"]
    else:
        modes = [args.mode]

    # Determine files to process
    if args.file:
        files = [Path(args.file)]
    else:
        files = get_sample_files()

    if not files:
        print("No sample files found. Run: python main.py --create-samples")
        return

    print(f"\nFiles to process: {len(files)}")
    print(f"Modes to compare: {modes}")

    # Run comparisons
    all_results = {}

    for file_path in files:
        print(f"\n{'#'*80}")
        print(f"# FILE: {file_path.name}")
        print(f"{'#'*80}")

        results = run_comparison(file_path, modes, verbose=args.verbose)
        all_results[file_path.name] = results

        print_comparison_table(file_path.name, results)

    # Print overall summary
    if len(files) > 1:
        print_summary(all_results)


if __name__ == "__main__":
    main()
