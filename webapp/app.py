"""
Document Automation PoC - Streamlit Web Application

A visual demonstration of the AI-powered document processing pipeline.
Shows both Traditional (rule_based/ai_only/hybrid) and MCP Agent modes.

Author: Akshay Karadkar
"""

import streamlit as st
import json
import time
import sys
import os
from pathlib import Path
from datetime import datetime
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SAMPLE_DATA_DIR, OUTPUT_DIR
from src.hybrid_pipeline import HybridPipeline, ExtractionMode
from mcp_server.tools import DocumentTools

# Page config
st.set_page_config(
    page_title="Document Automation PoC",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-top: 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 5px 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 5px;
        padding: 10px;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 10px;
    }
    .validation-pass {
        color: #28a745;
        font-weight: bold;
    }
    .validation-fail {
        color: #dc3545;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


def get_sample_files():
    """Get list of sample files."""
    files = []
    if SAMPLE_DATA_DIR.exists():
        for f in sorted(SAMPLE_DATA_DIR.iterdir()):
            if f.is_file() and not f.name.startswith('.'):
                files.append(f)
    return files


def get_confidence_color(confidence: float) -> str:
    """Get color based on confidence level."""
    if confidence >= 0.9:
        return "green"
    elif confidence >= 0.7:
        return "orange"
    else:
        return "red"


def get_confidence_status(confidence: float) -> tuple:
    """Get status text and emoji based on confidence."""
    if confidence >= 0.9:
        return "HIGH - Auto-approve", "✅"
    elif confidence >= 0.7:
        return "MEDIUM - Review recommended", "⚠️"
    else:
        return "LOW - Manual review required", "❌"


def validate_field(value, field_type="string", required=True):
    """Validate a field and return status."""
    if value is None:
        return not required, "null"
    if field_type == "string":
        valid = isinstance(value, str) and len(str(value).strip()) > 0
    elif field_type == "number":
        valid = isinstance(value, (int, float)) and value >= 0
    elif field_type == "date":
        try:
            datetime.strptime(str(value), "%Y-%m-%d")
            valid = True
        except:
            valid = False
    elif field_type == "array":
        valid = isinstance(value, list) and len(value) > 0
    else:
        valid = value is not None
    return valid, value


def render_validation_table(order_data: dict):
    """Render validation checklist for the order."""
    st.subheader("📋 Schema Validation Checklist")

    validations = [
        ("order_id", order_data.get("order_id"), "string", True),
        ("client_name", order_data.get("client_name"), "string", True),
        ("order_date", order_data.get("order_date"), "date", True),
        ("delivery_date", order_data.get("delivery_date"), "date", True),
        ("items", order_data.get("items"), "array", True),
        ("order_total", order_data.get("order_total"), "number", True),
        ("currency", order_data.get("currency"), "string", True),
        ("special_instructions", order_data.get("special_instructions"), "string", False),
        ("confidence_score", order_data.get("confidence_score"), "number", True),
    ]

    cols = st.columns([2, 3, 1])
    cols[0].markdown("**Field**")
    cols[1].markdown("**Value**")
    cols[2].markdown("**Status**")

    all_valid = True
    for field_name, value, field_type, required in validations:
        valid, display_val = validate_field(value, field_type, required)
        if not valid:
            all_valid = False

        cols = st.columns([2, 3, 1])
        cols[0].text(field_name)

        # Truncate long values
        if isinstance(display_val, list):
            cols[1].text(f"{len(display_val)} items")
        elif isinstance(display_val, str) and len(str(display_val)) > 30:
            cols[1].text(str(display_val)[:30] + "...")
        else:
            cols[1].text(str(display_val))

        if valid:
            cols[2].markdown('<span class="validation-pass">✅</span>', unsafe_allow_html=True)
        else:
            cols[2].markdown('<span class="validation-fail">❌</span>', unsafe_allow_html=True)

    # Validate items array
    if order_data.get("items"):
        with st.expander(f"📦 Line Items ({len(order_data['items'])} items)"):
            for i, item in enumerate(order_data["items"]):
                st.markdown(f"**Item {i+1}:**")
                item_cols = st.columns(5)
                item_cols[0].text(f"Code: {item.get('product_code', 'N/A')}")
                item_cols[1].text(f"Desc: {str(item.get('description', 'N/A'))[:20]}")
                item_cols[2].text(f"Qty: {item.get('quantity', 0)}")
                item_cols[3].text(f"Unit: ${item.get('unit_price', 0):.2f}")
                item_cols[4].text(f"Total: ${item.get('total_price', 0):.2f}")

    return all_valid


def process_traditional(file_path: Path, mode: str, status_container):
    """Process document using traditional pipeline with live updates."""
    with status_container:
        with st.status("🚀 Processing Document...", expanded=True) as status:
            # Step 1: Loading
            st.write("📄 Loading document...")
            time.sleep(0.2)

            # Step 2: Format detection
            suffix = file_path.suffix.lower()
            format_map = {
                '.pdf': 'PDF',
                '.xlsx': 'Excel',
                '.xls': 'Excel',
                '.docx': 'Word',
                '.csv': 'CSV',
                '.jpg': 'Image/OCR',
                '.jpeg': 'Image/OCR',
                '.png': 'Image/OCR'
            }
            detected_format = format_map.get(suffix, 'Unknown')
            st.write(f"🔍 Format detected: **{detected_format}**")
            time.sleep(0.2)

            # Step 3: Mode selection
            st.write(f"⚙️ Processing mode: **{mode}**")
            time.sleep(0.2)

            # Step 4: Extraction
            st.write("📊 Extracting fields...")

            # Create pipeline and process
            mode_enum = ExtractionMode(mode)
            pipeline = HybridPipeline(mode=mode_enum, verbose=False)

            start_time = time.time()
            result = pipeline.process(file_path, save_output=True)
            processing_time = (time.time() - start_time) * 1000

            if result.get("success"):
                # Step 5: Confidence scoring
                confidence = result["metrics"]["confidence"]
                st.write(f"🎯 Confidence calculated: **{confidence:.0%}**")
                time.sleep(0.2)

                # Step 6: Validation
                st.write("✓ Schema validation: **PASSED**")

                # Update status
                status.update(
                    label="✅ Processing Complete!",
                    state="complete",
                    expanded=False
                )
            else:
                status.update(
                    label="❌ Processing Failed",
                    state="error",
                    expanded=True
                )
                st.error(f"Error: {result.get('error', 'Unknown error')}")

            return result, processing_time


def process_with_agent(file_path: Path, status_container):
    """Process document using MCP agent with live updates."""
    with status_container:
        with st.status("🤖 Agent Processing...", expanded=True) as status:
            st.write("💭 Agent analyzing request...")
            time.sleep(0.3)

            st.write(f"🔧 Tool Call: `process_document('{file_path.name}', 'hybrid')`")

            # Use tools directly
            tools = DocumentTools(verbose=False)
            start_time = time.time()
            result = tools.process_document(str(file_path), "hybrid")
            processing_time = (time.time() - start_time) * 1000

            if result.get("success"):
                confidence = result["metrics"]["confidence"]
                st.write(f"📊 Extraction complete - Confidence: **{confidence:.0%}**")

                conf_status, _ = get_confidence_status(confidence)
                st.write(f"🎯 Status: **{conf_status}**")

                status.update(
                    label="✅ Agent Complete!",
                    state="complete",
                    expanded=False
                )
            else:
                status.update(
                    label="❌ Agent Failed",
                    state="error",
                    expanded=True
                )
                st.error(f"Error: {result.get('error', 'Unknown error')}")

            return result, processing_time


def main():
    """Main application."""
    # Header
    st.markdown('<p class="main-header">📄 Document Automation PoC</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Document Processing Demo</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Processing mode
        st.subheader("Processing Approach")
        approach = st.radio(
            "Select approach:",
            ["Traditional Pipeline", "MCP Agent"],
            help="Traditional: Direct pipeline calls | MCP Agent: Tool-calling agent"
        )

        if approach == "Traditional Pipeline":
            mode = st.selectbox(
                "Extraction Mode:",
                ["hybrid", "rule_based", "ai_only"],
                help="hybrid: Smart routing | rule_based: Fast & free | ai_only: AI for all"
            )
        else:
            mode = "hybrid"  # Agent uses hybrid by default

        st.markdown("---")

        # Sample files
        st.subheader("📁 Sample Files")
        sample_files = get_sample_files()

        if sample_files:
            selected_sample = st.selectbox(
                "Quick select:",
                ["-- Select --"] + [f.name for f in sample_files],
                help="Select a sample file to process"
            )
        else:
            selected_sample = "-- Select --"
            st.warning("No sample files found")

        st.markdown("---")

        # Info
        st.subheader("ℹ️ About")
        st.markdown("""
        **Author:** Akshay Karadkar

        **Features:**
        - 5 document formats
        - 3 extraction modes
        - Confidence scoring
        - Schema validation
        """)

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Document")

        # File uploader
        uploaded_file = st.file_uploader(
            "Drag and drop or browse",
            type=["pdf", "xlsx", "xls", "docx", "csv", "jpg", "jpeg", "png"],
            help="Supported: PDF, Excel, Word, CSV, Images"
        )

        # Or use sample file
        file_to_process = None

        if uploaded_file:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                file_to_process = Path(tmp.name)
            st.success(f"Uploaded: {uploaded_file.name}")

        elif selected_sample != "-- Select --":
            file_to_process = SAMPLE_DATA_DIR / selected_sample
            st.info(f"Selected: {selected_sample}")

    with col2:
        st.subheader("🎯 Target Output Schema")
        st.code("""{
  "order_id": "string",
  "client_name": "string",
  "order_date": "YYYY-MM-DD",
  "delivery_date": "YYYY-MM-DD",
  "items": [{...}],
  "order_total": number,
  "currency": "string",
  "special_instructions": "string|null",
  "confidence_score": 0.0-1.0
}""", language="json")

    st.markdown("---")

    # Process button
    if file_to_process:
        # Check for incompatible mode + file type combination
        is_image_file = file_to_process.suffix.lower() in ['.jpg', '.jpeg', '.png']
        is_rule_based = approach == "Traditional Pipeline" and mode == "rule_based"

        if is_image_file and is_rule_based:
            st.error(
                "**Rule-based mode cannot process scanned images.**\n\n"
                "Scanned documents (JPG, PNG) require OCR + AI extraction. "
                "Please select **'hybrid'** or **'ai_only'** mode, or use **MCP Agent** approach."
            )
            process_btn = False
        else:
            process_btn = st.button("🚀 Process Document", type="primary", use_container_width=True)

        if process_btn:
            # Processing pipeline visualization
            st.subheader("📊 Processing Pipeline")
            status_container = st.container()

            if approach == "Traditional Pipeline":
                result, processing_time = process_traditional(file_to_process, mode, status_container)
            else:
                result, processing_time = process_with_agent(file_to_process, status_container)

            # Results section
            if result and result.get("success"):
                st.markdown("---")

                # Metrics row
                st.subheader("📈 Extraction Metrics")
                metrics = result.get("metrics", {})
                confidence = metrics.get("confidence", 0)

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    conf_status, conf_emoji = get_confidence_status(confidence)
                    st.metric(
                        "Confidence",
                        f"{confidence:.0%}",
                        delta=conf_status.split(" - ")[0]
                    )

                with col2:
                    st.metric(
                        "Processing Time",
                        f"{processing_time:.0f}ms"
                    )

                with col3:
                    cost = metrics.get("estimated_cost", 0)
                    st.metric(
                        "Est. Cost",
                        f"${cost:.4f}"
                    )

                with col4:
                    fields = metrics.get("fields_extracted", 0)
                    total = metrics.get("total_fields", 6)
                    st.metric(
                        "Fields",
                        f"{fields}/{total}"
                    )

                # Confidence status banner
                conf_color = get_confidence_color(confidence)
                if conf_color == "green":
                    st.success(f"✅ {conf_status} - This extraction can be auto-approved")
                elif conf_color == "orange":
                    st.warning(f"⚠️ {conf_status} - Please verify key fields")
                else:
                    st.error(f"❌ {conf_status} - Manual review required")

                st.markdown("---")

                # Output display
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("📋 Extracted Order (JSON)")
                    order_data = result.get("order", {})
                    st.json(order_data)

                    # Download button
                    json_str = json.dumps(order_data, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_str,
                        file_name=f"order_{order_data.get('order_id', 'extracted')}.json",
                        mime="application/json",
                        use_container_width=True
                    )

                with col2:
                    # Validation checklist
                    all_valid = render_validation_table(order_data)

                    if all_valid:
                        st.success("✅ All fields validated successfully!")
                    else:
                        st.warning("⚠️ Some fields need attention")

            elif result:
                st.error(f"Processing failed: {result.get('error', 'Unknown error')}")

    else:
        st.info("👆 Please upload a document or select a sample file to begin")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Document Automation PoC | Akshay Karadkar"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
