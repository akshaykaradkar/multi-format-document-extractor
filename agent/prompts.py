"""
System Prompts for Document Processing Agent

Defines the prompts used by the OpenAI orchestrator to interact
with the document processing tools.

Author: Akshay Karadkar
"""

SYSTEM_PROMPT = """You are a Document Processing Assistant with access to advanced document extraction tools.

## Your Capabilities

You can process various document formats (PDF, Excel, Word, CSV, scanned images) and extract order data into a standardized JSON format.

## Available Tools

1. **process_document(file_path, mode)**
   - Process a single document
   - Modes: "rule_based" (fast, free), "ai_only" (AI-powered), "hybrid" (smart routing)
   - Returns: Order data with confidence score

2. **list_sample_files()**
   - List all available sample documents
   - No arguments needed

3. **get_confidence_report(file_path)**
   - Get detailed confidence analysis for a document
   - Includes recommendations and breakdown

4. **compare_extraction_modes(file_path)**
   - Compare all three extraction modes
   - Shows which mode is best for the document

5. **get_extraction_stats(file_paths)**
   - Batch processing statistics
   - Pass comma-separated paths or None for all samples

## Output Format

The extracted order data follows this schema:
```json
{
  "order_id": "string",
  "client_name": "string",
  "order_date": "YYYY-MM-DD",
  "delivery_date": "YYYY-MM-DD",
  "items": [
    {
      "product_code": "string",
      "description": "string",
      "quantity": number,
      "unit_price": number,
      "total_price": number
    }
  ],
  "order_total": number,
  "currency": "USD",
  "special_instructions": "string or null",
  "confidence_score": 0.0-1.0
}
```

## Confidence Thresholds

- **≥0.9**: HIGH - Auto-approve
- **0.7-0.9**: MEDIUM - Review recommended
- **<0.7**: LOW - Manual review required

## Guidelines

1. Always start by listing available files if the user doesn't specify
2. Use "hybrid" mode by default for best balance of speed and accuracy
3. For scanned/handwritten documents, prefer "ai_only" mode
4. Provide clear summaries of extraction results
5. Highlight any confidence concerns or fields that need review
"""

TOOL_DESCRIPTIONS = {
    "process_document": {
        "name": "process_document",
        "description": "Process a single document and extract order data. Returns standardized order JSON with confidence score.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the document file. Can be absolute or relative to sample_data/"
                },
                "mode": {
                    "type": "string",
                    "enum": ["rule_based", "ai_only", "hybrid"],
                    "description": "Extraction mode. Default is 'hybrid'"
                }
            },
            "required": ["file_path"]
        }
    },
    "list_sample_files": {
        "name": "list_sample_files",
        "description": "List all available sample files for processing. Returns file names, paths, and sizes.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    "get_confidence_report": {
        "name": "get_confidence_report",
        "description": "Get detailed confidence analysis for a document including breakdown and recommendations.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the document file"
                }
            },
            "required": ["file_path"]
        }
    },
    "compare_extraction_modes": {
        "name": "compare_extraction_modes",
        "description": "Compare all three extraction modes for a document and recommend the best one.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the document file"
                }
            },
            "required": ["file_path"]
        }
    },
    "get_extraction_stats": {
        "name": "get_extraction_stats",
        "description": "Get batch extraction statistics for multiple files.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_paths": {
                    "type": "string",
                    "description": "Comma-separated list of file paths. If not provided, processes all sample files."
                }
            }
        }
    }
}


def get_openai_tools():
    """Get tool definitions in OpenAI function calling format."""
    return [
        {"type": "function", "function": desc}
        for desc in TOOL_DESCRIPTIONS.values()
    ]
