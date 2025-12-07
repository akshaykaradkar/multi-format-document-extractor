"""
Run Streamlit Web Application

Entry point for the Document Automation PoC visual demo.
Launches the Streamlit webapp for document processing demonstration.

Usage:
    streamlit run run_webapp.py

Author: Akshay Karadkar
"""

import subprocess
import sys
from pathlib import Path

# Get the webapp app.py path
WEBAPP_PATH = Path(__file__).parent / "webapp" / "app.py"


def main():
    """Launch the Streamlit webapp."""
    if not WEBAPP_PATH.exists():
        print(f"Error: Webapp not found at {WEBAPP_PATH}")
        sys.exit(1)

    print("=" * 60)
    print("  Document Automation PoC - Visual Demo")
    print("  Akshay Karadkar")
    print("=" * 60)
    print(f"\nLaunching Streamlit webapp...")
    print(f"App path: {WEBAPP_PATH}")
    print("\nOpen in browser: http://localhost:8501")
    print("Press Ctrl+C to stop the server\n")

    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(WEBAPP_PATH),
        "--server.port=8501",
        "--server.headless=false",
        "--browser.gatherUsageStats=false"
    ])


if __name__ == "__main__":
    main()
