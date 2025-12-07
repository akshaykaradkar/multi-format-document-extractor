#!/usr/bin/env python3
"""
MCP Document Processing Demo

Demonstrates the MCP server and OpenAI agent for document processing.
This shows the advanced agent-based architecture built on top of the
existing HybridPipeline.

Author: Akshay Karadkar
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def demo_tools_direct():
    """Demo 1: Using tools directly (no LLM)."""
    print("\n" + "=" * 70)
    print("DEMO 1: Direct Tool Usage (No LLM Cost)")
    print("=" * 70)

    from mcp_server.tools import DocumentTools

    tools = DocumentTools(verbose=False)

    # List files
    print("\n1. Listing available sample files...")
    files = tools.list_sample_files()
    print(f"   Found {files['file_count']} files in {files['directory']}")
    for f in files['files']:
        print(f"   - {f['name']} ({f['size_kb']} KB)")

    # Process first file
    if files['files']:
        print(f"\n2. Processing {files['files'][0]['name']}...")
        result = tools.process_document(
            files['files'][0]['path'],
            mode="hybrid"
        )

        if result['success']:
            order = result['order']
            metrics = result['metrics']
            print(f"   Order ID: {order['order_id']}")
            print(f"   Client: {order['client_name']}")
            print(f"   Total: ${order['order_total']}")
            print(f"   Confidence: {metrics['confidence']:.2f}")
            print(f"   Status: {result['confidence_status']}")
        else:
            print(f"   Error: {result.get('error')}")

    # Get stats for all files
    print("\n3. Batch processing all files...")
    stats = tools.get_extraction_stats()
    summary = stats['summary']
    print(f"   Files processed: {summary['total_files']}")
    print(f"   Success rate: {summary['success_rate']}")
    print(f"   Total time: {summary['total_processing_time_ms']:.0f}ms")
    print(f"   Avg time: {summary['avg_processing_time_ms']:.0f}ms")
    print(f"   Total cost: ${summary['total_estimated_cost_usd']:.4f}")


def demo_agent():
    """Demo 2: Using the OpenAI agent with natural language."""
    print("\n" + "=" * 70)
    print("DEMO 2: OpenAI Agent (Natural Language Interface)")
    print("=" * 70)

    try:
        from agent import DocumentAgent
    except ImportError as e:
        print(f"\nError importing agent: {e}")
        print("Make sure openai is installed: pip install openai")
        return

    try:
        agent = DocumentAgent(verbose=True)
    except RuntimeError as e:
        print(f"\nError: {e}")
        print("Set OPENAI_API_KEY in .env file to use the agent.")
        return

    # Demo queries
    queries = [
        "What files are available for processing?",
        "Process the PDF file using hybrid mode and show me the results",
        "Compare all extraction modes for the Excel file",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'─'*70}")
        print(f"Query {i}: {query}")
        print(f"{'─'*70}")

        try:
            response = agent.process(query)
            # Response is already printed by the agent in verbose mode
        except Exception as e:
            print(f"Error: {e}")

        # Reset between queries for demo
        agent.reset()


def demo_comparison():
    """Demo 3: Compare extraction modes."""
    print("\n" + "=" * 70)
    print("DEMO 3: Extraction Mode Comparison")
    print("=" * 70)

    from mcp_server.tools import DocumentTools

    tools = DocumentTools(verbose=False)
    files = tools.list_sample_files()

    if not files['files']:
        print("No sample files found!")
        return

    # Pick the PDF file for comparison
    pdf_file = None
    for f in files['files']:
        if f['extension'] == '.pdf':
            pdf_file = f['path']
            break

    if not pdf_file:
        pdf_file = files['files'][0]['path']

    print(f"\nComparing modes for: {Path(pdf_file).name}")
    print("-" * 50)

    comparison = tools.compare_extraction_modes(pdf_file)

    for mode in ['rule_based', 'ai_only', 'hybrid']:
        result = comparison.get(mode, {})
        if result.get('success'):
            print(f"\n{mode.upper()}:")
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  Time: {result['processing_time_ms']:.0f}ms")
            print(f"  Cost: ${result['estimated_cost_usd']:.4f}")
            print(f"  Method: {result['method_used']}")
        else:
            print(f"\n{mode.upper()}: Failed - {result.get('error')}")

    if 'comparison' in comparison:
        print(f"\n{'='*50}")
        print("RECOMMENDATION:")
        print(f"  Best Confidence: {comparison['comparison']['best_confidence']}")
        print(f"  Fastest: {comparison['comparison']['fastest']}")
        print(f"  Cheapest: {comparison['comparison']['cheapest']}")
        print(f"  Suggested: {comparison['comparison']['recommendation']}")


def demo_interactive():
    """Demo 4: Interactive agent mode."""
    print("\n" + "=" * 70)
    print("DEMO 4: Interactive Agent Mode")
    print("=" * 70)
    print("\nType your questions in natural language.")
    print("Commands: 'quit' to exit, 'reset' to clear history")
    print("-" * 70)

    try:
        from agent import DocumentAgent
        agent = DocumentAgent(verbose=True)
    except Exception as e:
        print(f"Error: {e}")
        return

    while True:
        try:
            query = input("\nYou: ").strip()

            if not query:
                continue
            if query.lower() == 'quit':
                print("Goodbye!")
                break
            if query.lower() == 'reset':
                agent.reset()
                print("History cleared.")
                continue

            agent.process(query)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("MCP DOCUMENT PROCESSING - DEMONSTRATION")
    print("=" * 70)
    print("""
This demo showcases the MCP (Model Context Protocol) server architecture
built on top of the existing HybridPipeline document processing system.

Architecture:
  User -> OpenAI Agent -> MCP Tools -> HybridPipeline -> Output

Available demos:
  1. Direct Tool Usage (no LLM cost)
  2. OpenAI Agent (natural language)
  3. Extraction Mode Comparison
  4. Interactive Mode
""")

    if len(sys.argv) > 1:
        demo_num = sys.argv[1]
        if demo_num == "1":
            demo_tools_direct()
        elif demo_num == "2":
            demo_agent()
        elif demo_num == "3":
            demo_comparison()
        elif demo_num == "4":
            demo_interactive()
        else:
            print(f"Unknown demo: {demo_num}")
            print("Usage: python mcp_demo.py [1|2|3|4]")
    else:
        # Run all non-interactive demos
        demo_tools_direct()
        demo_comparison()

        # Ask if user wants agent demo
        print("\n" + "-" * 70)
        response = input("Run OpenAI Agent demo? (requires API key) [y/N]: ")
        if response.lower() == 'y':
            demo_agent()

        # Ask if user wants interactive mode
        print("\n" + "-" * 70)
        response = input("Enter interactive mode? [y/N]: ")
        if response.lower() == 'y':
            demo_interactive()


if __name__ == "__main__":
    main()
