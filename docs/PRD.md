# Product Requirements Document (PRD)
# Document Automation & Data Harmonization System

**Project:** AI-Powered Purchase Order Processing
**Author:** Akshay Karadkar
**Date:** December 7, 2024
**Version:** 1.0

---

## 1. Executive Summary

### 1.1 Problem Statement
The company receives purchase orders from 25+ clients in various formats (PDF invoices, Excel spreadsheets, Word documents, CSV files, and handwritten/scanned forms). These orders must be processed by a single vendor requiring data in a standardized JSON format. The current manual process leads to:
- Significant delays in order processing
- Human errors in data entry
- Inconsistent data quality
- High operational costs (500-800 orders/month)

### 1.2 Solution Overview
An AI-powered document automation system that:
- Automatically ingests documents in 5+ formats
- Extracts order data using hybrid parsing (local libraries + LLM)
- Transforms and harmonizes data into a standardized JSON schema
- Provides confidence scoring for quality assurance
- Supports human-in-the-loop review for edge cases

### 1.3 Business Value
- **Time Savings:** Reduce processing time from ~15 min/order to <1 min/order
- **Error Reduction:** Eliminate manual data entry errors
- **Scalability:** Handle volume growth without proportional staff increase
- **Consistency:** Ensure uniform output format for vendor integration

---

## 2. Scope & Objectives

### 2.1 In Scope (MVP)
| Feature | Priority | Description |
|---------|----------|-------------|
| PDF Parsing | P0 | Extract data from structured PDF invoices |
| Excel Parsing | P0 | Process multi-sheet Excel workbooks |
| Word Parsing | P0 | Handle Word documents with tables and free text |
| CSV Parsing | P0 | Parse CSV files with varying column structures |
| OCR/Scanned | P0 | Extract data from scanned/handwritten forms |
| JSON Output | P0 | Generate standardized JSON per schema |
| Confidence Scoring | P1 | Provide extraction confidence metrics |
| Error Handling | P1 | Graceful handling of edge cases |

### 2.2 Out of Scope (Future)
- Real-time streaming ingestion
- Multi-language document support
- Custom training/fine-tuning
- Full production deployment infrastructure
- User authentication/authorization

### 2.3 Success Criteria
1. Successfully parse all 5 document formats
2. Output conforms to exact JSON schema specification
3. Confidence scores accurately reflect extraction quality
4. End-to-end processing completes without manual intervention
5. Demo works with real API calls (not mocked)

---

## 3. User Stories & Requirements

### 3.1 Primary User Stories

**US-001: Process PDF Invoice**
> As a data processor, I want to upload a PDF invoice and receive structured JSON output, so that I can forward it to the vendor system without manual data entry.

**Acceptance Criteria:**
- System accepts PDF file input
- Extracts order ID, client name, dates, line items, totals
- Outputs valid JSON matching target schema
- Includes confidence score for the extraction

**US-002: Process Multi-Sheet Excel**
> As a data processor, I want to upload an Excel workbook with data spread across multiple sheets, so that all order information is consolidated into a single JSON output.

**Acceptance Criteria:**
- System reads all relevant sheets (Order_Info, Line_Items, Notes)
- Maps custom field names to standard schema fields
- Handles varying column structures
- Consolidates into unified JSON output

**US-003: Process Word Document**
> As a data processor, I want to upload a Word document with mixed formatting, so that order data is extracted regardless of document structure.

**Acceptance Criteria:**
- Extracts data from embedded tables
- Parses free-text order information
- Handles inconsistent formatting
- Produces standardized JSON output

**US-004: Process CSV File**
> As a data processor, I want to upload a CSV file with varying column orders, so that the system intelligently maps fields to the target schema.

**Acceptance Criteria:**
- Handles different column orderings
- Normalizes mixed date formats
- Processes optional/missing fields gracefully
- Groups multi-line orders correctly

**US-005: Process Scanned Form**
> As a data processor, I want to upload a scanned or handwritten order form, so that OCR extracts the data into structured JSON.

**Acceptance Criteria:**
- Performs OCR on image/scanned PDF
- Extracts handwritten text accurately
- Identifies form fields (checkboxes, lines)
- Outputs JSON with appropriate confidence score

**US-006: Review Low-Confidence Extractions**
> As a data processor, I want to see confidence scores for each extraction, so that I can prioritize manual review for uncertain results.

**Acceptance Criteria:**
- Confidence score (0-1) included in output
- Scores ≥0.9 flagged as auto-approve
- Scores 0.7-0.9 flagged for review
- Scores <0.7 flagged for manual processing

---

## 4. Functional Requirements

### 4.1 Document Ingestion
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-001 | Accept PDF files (.pdf) | P0 |
| FR-002 | Accept Excel files (.xlsx, .xls) | P0 |
| FR-003 | Accept Word files (.docx) | P0 |
| FR-004 | Accept CSV files (.csv) | P0 |
| FR-005 | Accept image files (.jpg, .png) for OCR | P0 |
| FR-006 | Auto-detect file format from extension | P0 |
| FR-007 | Validate file is readable before processing | P1 |

### 4.2 Data Extraction
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-010 | Extract order ID/number | P0 |
| FR-011 | Extract client/company name | P0 |
| FR-012 | Extract order date | P0 |
| FR-013 | Extract delivery/need-by date | P0 |
| FR-014 | Extract line items (product, qty, price) | P0 |
| FR-015 | Extract order total | P0 |
| FR-016 | Extract special instructions/notes | P1 |
| FR-017 | Infer currency from context | P1 |

### 4.3 Data Transformation
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-020 | Normalize dates to YYYY-MM-DD format | P0 |
| FR-021 | Map client-specific field names to schema | P0 |
| FR-022 | Calculate line item totals if missing | P1 |
| FR-023 | Validate order total matches line items | P1 |
| FR-024 | Generate product codes if missing | P2 |

### 4.4 Output Generation
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-030 | Output valid JSON per target schema | P0 |
| FR-031 | Include confidence_score field | P0 |
| FR-032 | Save output to file system | P0 |
| FR-033 | Pretty-print JSON for readability | P1 |

---

## 5. Non-Functional Requirements

### 5.1 Performance
| ID | Requirement | Target |
|----|-------------|--------|
| NFR-001 | PDF processing time | <5 seconds |
| NFR-002 | Excel processing time | <3 seconds |
| NFR-003 | Word processing time | <3 seconds |
| NFR-004 | CSV processing time | <1 second |
| NFR-005 | OCR processing time | <10 seconds |

### 5.2 Reliability
| ID | Requirement | Target |
|----|-------------|--------|
| NFR-010 | Extraction accuracy (structured docs) | >95% |
| NFR-011 | Extraction accuracy (OCR) | >85% |
| NFR-012 | System availability | N/A (PoC) |

### 5.3 Maintainability
| ID | Requirement | Description |
|----|-------------|-------------|
| NFR-020 | Modular architecture | Each parser is independent |
| NFR-021 | Configuration-driven | API keys, thresholds in config |
| NFR-022 | Extensible design | Easy to add new document types |

---

## 6. Target JSON Schema

All processed orders MUST conform to this schema:

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
      "quantity": "number",
      "unit_price": "number",
      "total_price": "number"
    }
  ],
  "order_total": "number",
  "currency": "string",
  "special_instructions": "string | null",
  "confidence_score": "number (0.0 - 1.0)"
}
```

### 6.1 Field Specifications

| Field | Type | Required | Validation |
|-------|------|----------|------------|
| order_id | string | Yes | Non-empty |
| client_name | string | Yes | Non-empty |
| order_date | string | Yes | YYYY-MM-DD format |
| delivery_date | string | Yes | YYYY-MM-DD format |
| items | array | Yes | At least 1 item |
| items[].product_code | string | Yes | Non-empty |
| items[].description | string | Yes | Non-empty |
| items[].quantity | number | Yes | Positive integer |
| items[].unit_price | number | Yes | Non-negative |
| items[].total_price | number | Yes | Non-negative |
| order_total | number | Yes | Non-negative |
| currency | string | Yes | Default "USD" |
| special_instructions | string | No | Nullable |
| confidence_score | number | Yes | 0.0 to 1.0 |

---

## 7. Test Data & Client Scenarios

### 7.1 Client A - TechCorp Solutions (PDF)
- **Format:** Clean PDF with tabular layout
- **Characteristics:** Consistent structure, clear field labels
- **Order ID:** PO-2024-1247
- **Items:** Widget Pro (50 @ $25), Gadget Max (25 @ $45)
- **Total:** $2,375.00
- **Notes:** Rush delivery required

### 7.2 Client B - Global Manufacturing Inc (Excel)
- **Format:** Multi-sheet Excel workbook
- **Characteristics:** Custom field names, data across 3 sheets
- **Sheets:** Order_Info, Line_Items, Notes
- **Challenge:** Field mapping from custom names

### 7.3 Client C - Regional Distributors (Word)
- **Format:** Word document with mixed formatting
- **Characteristics:** Embedded tables, free-text sections
- **Order ID:** RD-240815-A
- **Items:** Industrial Pump X200 (3 @ $850), Filter Cartridge Set (12 @ $45)
- **Notes:** Phoenix warehouse, delivery before 3 PM

### 7.4 Client D - Supply Chain Partners (CSV)
- **Format:** CSV with varying column structures
- **Characteristics:** Different column orders, mixed date formats
- **Order ID:** SCP-2024-0445
- **Items:** Heavy Duty Clamp (100 @ $12.50), Mounting Bracket (75 @ $8.25)
- **Challenge:** Multi-row order grouping

### 7.5 Client E - Local Hardware Co (Scanned)
- **Format:** Scanned PDF of handwritten/typed form
- **Characteristics:** OCR required, potential quality issues
- **Order ID:** LHC-240318
- **Items:** Screws Phillips Head #8 x 1" (500 @ $0.15), Wood Stain Oak (6 gal @ $28.00)
- **Notes:** Rush delivery, call before delivery
- **Contact:** Mike Johnson (555) 123-4567

---

## 8. Technical Approach

### 8.1 Hybrid Processing Strategy
```
┌─────────────────────────────────────────────────────────┐
│                    Document Input                        │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│               Format Detection (by extension)            │
└─────────────────────────┬───────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐
│ Local Parser│  │ Local Parser│  │ LLM Parser (GPT-4o) │
│ (PDF/Excel/ │  │    (CSV)    │  │ (Scanned/Complex)   │
│  Word)      │  │             │  │                     │
└──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘
       │                │                     │
       └────────────────┼─────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│            Data Transformer & Field Normalizer           │
│  - Date normalization (YYYY-MM-DD)                       │
│  - Field mapping (custom → standard)                     │
│  - Total calculation/validation                          │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│              Schema Validator (Pydantic)                 │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│                 Confidence Scorer                        │
│  - Field completeness (40%)                              │
│  - Extraction source confidence (40%)                    │
│  - Validation pass rate (20%)                            │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│              Standardized JSON Output                    │
└─────────────────────────────────────────────────────────┘
```

### 8.2 Technology Stack
| Component | Technology | Rationale |
|-----------|------------|-----------|
| PDF Parsing | pdfplumber | Best for tables, structured PDFs |
| Excel Parsing | pandas + openpyxl | Multi-sheet support, DataFrame conversion |
| Word Parsing | python-docx | Native table and paragraph extraction |
| CSV Parsing | pandas | Flexible, handles varying structures |
| OCR | GPT-4o Vision | Best accuracy for handwritten content |
| Field Extraction | OpenAI Structured Outputs | Guaranteed JSON compliance |
| Validation | Pydantic v2 | Type safety, automatic validation |
| Configuration | python-dotenv | Environment variable management |

---

## 9. Confidence Scoring Algorithm

### 9.1 Score Components
| Component | Weight | Description |
|-----------|--------|-------------|
| Field Completeness | 40% | % of required fields populated |
| Source Confidence | 40% | Parser/OCR confidence level |
| Validation Score | 20% | Schema validation pass rate |

### 9.2 Calculation
```
confidence_score = (
    field_completeness * 0.4 +
    source_confidence * 0.4 +
    validation_score * 0.2
)
```

### 9.3 Thresholds & Actions
| Score Range | Status | Action |
|-------------|--------|--------|
| ≥ 0.90 | High Confidence | Auto-approve for processing |
| 0.70 - 0.89 | Medium Confidence | Flag for optional review |
| < 0.70 | Low Confidence | Require manual review |

---

## 10. Deliverables

### 10.1 Code Deliverables
| Deliverable | Description |
|-------------|-------------|
| src/ | Source code with modular parser architecture |
| sample_data/ | Mock files for all 5 client scenarios |
| tests/ | Basic test coverage |
| main.py | Demo script showing end-to-end processing |
| requirements.txt | Python dependencies |

### 10.2 Documentation Deliverables
| Deliverable | Weight | Description |
|-------------|--------|-------------|
| TECHNICAL_ARCHITECTURE.md | 40% | System design and components |
| IMPLEMENTATION_STRATEGY.md | 25% | Roadmap and phasing |
| BUSINESS_CASE.md | 15% | ROI and risk assessment |
| PRESENTATION.md | N/A | 5-7 slide summary |
| README.md | N/A | Setup and usage guide |

---

## 11. Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| OCR accuracy on poor quality scans | High | Medium | Confidence scoring + human review |
| New document formats from clients | Medium | High | Extensible parser architecture |
| LLM hallucination on ambiguous data | High | Medium | Validation + confidence thresholds |
| API rate limits/costs | Medium | Low | Hybrid approach (local first) |
| Date format variations | Low | High | Robust date parsing library |

---

## 12. Future Considerations

### 12.1 Phase 2 Enhancements
- Support for additional document formats (PDF forms, emails)
- Multi-language document support
- Batch processing capabilities
- Web-based user interface
- Integration with vendor APIs

### 12.2 Production Readiness
- Kubernetes deployment configuration
- Monitoring and alerting
- Audit logging
- User authentication
- Database persistence for processed orders

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| OCR | Optical Character Recognition |
| LLM | Large Language Model |
| HITL | Human-in-the-Loop |
| IDP | Intelligent Document Processing |
| PoC | Proof of Concept |
| MVP | Minimum Viable Product |

---

## Appendix B: References

- [OpenAI Structured Outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/)
- [GPT-4o Vision Documentation](https://platform.openai.com/docs/guides/vision)
- [pdfplumber Documentation](https://github.com/jsvine/pdfplumber)
- [AWS IDP Best Practices](https://aws.amazon.com/blogs/machine-learning/scalable-intelligent-document-processing-using-amazon-bedrock-data-automation/)
- [Confidence Scoring in Document AI](https://www.infrrd.ai/blog/confidence-scores-in-llms)
